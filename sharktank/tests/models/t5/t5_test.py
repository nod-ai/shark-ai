# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from copy import copy
from transformers.models.t5.modeling_t5 import (
    T5Attention as ReferenceT5Attention,
    T5LayerSelfAttention as ReferenceT5LayerSelfAttention,
    T5LayerFF as ReferenceT5LayerFF,
)
from transformers import (
    AutoTokenizer,
    T5EncoderModel as ReferenceT5EncoderModel,
    T5Config as ReferenceT5Config,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
import iree.runtime
from typing import Callable, Optional
import os
from collections import OrderedDict
import pytest
import torch
from torch.utils._pytree import tree_map
from unittest import TestCase
from parameterized import parameterized
from amdsharktank.types import (
    Theta,
    DefaultPrimitiveTensor,
    unbox_tensor,
    Dataset,
)
from amdsharktank.models.t5 import (
    T5Attention,
    T5SelfAttention,
    T5Config,
    T5Encoder,
    T5LayerFF,
    export_encoder_mlir,
    import_encoder_dataset_from_hugging_face,
)
from amdsharktank.models.t5.testing import (
    get_t5_encoder_toy_config,
    covert_t5_encoder_to_hugging_face,
    make_t5_encoder_random_theta,
)
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.random import make_rand_torch, make_random_mask
from amdsharktank.utils.testing import (
    assert_tensor_close,
    assert_text_encoder_state_close,
    skip,
    TempDirTestBase,
    get_test_prompts,
)
from amdsharktank.utils.hf_datasets import get_dataset
from amdsharktank.utils.iree import (
    get_iree_compiler_flags_from_object,
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from amdsharktank.transforms.dataset import set_float_dtype
from amdsharktank import ops
import iree.compiler

with_t5_data = pytest.mark.skipif("not config.getoption('with_t5_data')")

logger = get_logger(__name__)


def assert_t5_encoder_state_close(actual: torch.Tensor, expected: torch.Tensor):
    if actual.dtype == torch.bfloat16:
        # For both
        # testCompareV1_1XxlTorchEagerBf16AgainstHuggingFaceF32 and
        # testCompareV1_1XxlTorchEagerHuggingFaceBf16AgainstF32
        # the observed absolute numerical error is 0.610.
        # The error seems high as it corresponds to ~67° angular difference.
        # The majority of tokens have an error less than 0.02.
        worst_observed_cosine_similarity_per_token = 0.610
        tolerance_from_observed = 1.5
        atol = worst_observed_cosine_similarity_per_token * tolerance_from_observed
        worst_observed_outliers_fraction = 0.0207
        max_outliers_fraction = (
            worst_observed_outliers_fraction * tolerance_from_observed
        )
        assert_text_encoder_state_close(
            actual,
            expected,
            atol=atol,
            max_outliers_fraction=max_outliers_fraction,
            inlier_atol=0.02,
        )
    elif actual.dtype == torch.float32:
        assert_text_encoder_state_close(
            actual,
            expected,
            atol=1e-5,
        )
    else:
        raise ValueError(f"Unsupported actual dtype {actual.dtype}.")


class T5EncoderEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    def compare_torch_eager_vs_hugging_face(
        self,
        target_model: T5Encoder,
        input_kwargs: dict[str, torch.Tensor],
        assert_close: Callable[[torch.Tensor, torch.Tensor], None],
        reference_model: ReferenceT5EncoderModel | None = None,
        reference_dtype: torch.dtype | None = None,
    ):
        if reference_model is None:
            reference_model = covert_t5_encoder_to_hugging_face(target_model)
        if reference_dtype is not None:
            reference_model = reference_model.to(dtype=reference_dtype)

        logger.info("Invoking reference HuggingFace model...")
        reference_results = reference_model(**input_kwargs)
        logger.info("Invoking Torch eager model...")
        target_results = target_model(**input_kwargs)
        logger.info("Comparing outputs...")
        assert_close(
            target_results["last_hidden_state"],
            reference_results["last_hidden_state"],
        )

    def runTestV1_1CompareTorchEagerHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)
        reference_model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )
        reference_model.eval()

        target_model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=target_dtype
        )

        input_ids = tokenizer(
            get_test_prompts(),
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).input_ids

        def assert_close(actual: torch.Tensor, expected: torch.Tensor):
            assert_t5_encoder_state_close(
                actual,
                expected,
            )

        self.compare_torch_eager_vs_hugging_face(
            target_model=target_model,
            reference_model=reference_model,
            input_kwargs={"input_ids": input_ids},
            assert_close=assert_close,
        )

    def runTestV1_1CompareTorchEagerAgainstHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)
        reference_model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )
        reference_model.eval()

        dataset = import_encoder_dataset_from_hugging_face(huggingface_repo_id)
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )
        config = T5Config.from_properties(
            dataset.properties,
        )

        input_ids = tokenizer(
            get_test_prompts(),
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).input_ids

        target_model = T5Encoder(theta=dataset.root_theta, config=config)

        def assert_close(actual: torch.Tensor, expected: torch.Tensor):
            assert_t5_encoder_state_close(
                actual,
                expected,
            )

        self.compare_torch_eager_vs_hugging_face(
            target_model=target_model,
            reference_model=reference_model,
            input_kwargs={"input_ids": input_ids},
            assert_close=assert_close,
        )

    @parameterized.expand(
        [
            (torch.float32, torch.float64, 1e-5, 1e-5),
            (torch.bfloat16, torch.float64, 2e-2, 1e-2),
        ]
    )
    def testCompareToyEagerVsHuggigFace(
        self,
        target_dtype: torch.dtype,
        reference_dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        config = get_t5_encoder_toy_config()
        theta = make_t5_encoder_random_theta(config, dtype=target_dtype)
        target_model = T5Encoder(theta=theta, config=config)
        _, input_kwargs = target_model.sample_inputs(batch_size=2)

        def assert_close(actual: torch.Tensor, expected: torch.Tensor):
            assert_tensor_close(
                actual.to(dtype=expected.dtype), expected, atol=atol, rtol=rtol
            )

        self.compare_torch_eager_vs_hugging_face(
            target_model=target_model,
            input_kwargs=input_kwargs,
            assert_close=assert_close,
            reference_dtype=reference_dtype,
        )

    @skip
    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallTorchEagerHuggingFaceBf16AgainstF32(self):
        """Hugging Face model tests to estimate numerical error baseline for reference.
        We don't want to run this test regularly, but we would like to keep it around
        as a reference. It provides some baseline of what numerical error to expect.
        """
        self.runTestV1_1CompareTorchEagerHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )

    @skip
    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlTorchEagerHuggingFaceBf16AgainstF32(self):
        """Hugging Face model tests to estimate numerical error baseline for reference.
        We don't want to run this test regularly, but we would like to keep it around
        as a reference. It provides some baseline of what numerical error to expect.
        """
        self.runTestV1_1CompareTorchEagerHuggingFace(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallTorchEagerF32AgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallTorchEagerBf16AgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallTorchEagerBf16AgainstHuggingFaceBf16(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.bfloat16,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlTorchEagerF32AgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlTorchEagerBf16AgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )


@pytest.mark.usefixtures("caching", "iree_flags", "path_prefix")
class T5EncoderIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        if self.path_prefix is None:
            self.path_prefix = f"{self._temp_dir}/"

    def compare_iree_vs_eager(
        self,
        target_model: T5Encoder,
        input_kwargs: dict[str, torch.Tensor],
        assert_close: Callable[[torch.Tensor, torch.Tensor], None],
        reference_model: T5Encoder | None = None,
        reference_dtype: torch.dtype | None = None,
    ):
        if reference_model is None:
            reference_model = target_model
        if reference_dtype is not None:
            reference_theta = reference_model.theta.transform(
                functools.partial(set_float_dtype, dtype=reference_dtype)
            )
            reference_model = T5Encoder(
                theta=reference_theta, config=reference_model.config
            )

        batch_size = input_kwargs["input_ids"].shape[0]

        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            kwargs=input_kwargs,
            trace_path_prefix=f"{self.path_prefix}torch_",
        )
        reference_result = flatten_for_iree_signature(reference_result_dict)

        parameters_path = f"{self.path_prefix}parameters.irpa"
        if not self.caching or not os.path.exists(parameters_path):
            target_dataset = Dataset(
                properties=target_model.config.to_properties(),
                root_theta=target_model.theta,
            )
            target_dataset.save(parameters_path)

        mlir_path = f"{self.path_prefix}model.mlir"
        if not self.caching or not os.path.exists(mlir_path):
            logger.info("Exporting T5 encoder model to MLIR...")
            export_encoder_mlir(
                target_model,
                batch_sizes=[batch_size],
                mlir_output_path=mlir_path,
                dynamic_context_length=False,
            )
        iree_module_path = f"{self.path_prefix}model.vmfb"
        if not self.caching or not os.path.exists(iree_module_path):
            logger.info("Compiling MLIR file...")
            iree.compiler.compile_file(
                mlir_path,
                output_file=iree_module_path,
                extra_args=get_iree_compiler_flags_from_object(self),
            )

        iree_devices = get_iree_devices(driver=self.iree_device, device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            logger.info("Loading IREE module...")
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=parameters_path,
            )
            iree_args = prepare_iree_module_function_args(
                args=flatten_for_iree_signature(input_kwargs), devices=iree_devices
            )
            logger.info("Invoking IREE function...")
            iree_result = iree_to_torch(
                *run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name=f"forward_bs{batch_size}",
                    trace_path_prefix=f"{self.path_prefix}iree_",
                )
            )
            return [t.clone() for t in iree_result]

        iree_result = with_iree_device_context(run_iree_module, iree_devices)

        logger.info("Comparing outputs...")
        reference_result_last_hidden_state = reference_result[0]
        iree_result_last_hidden_state = iree_result[0]

        assert_close(iree_result_last_hidden_state, reference_result_last_hidden_state)

    def runTestV1_1CompareIreeAgainstTorchEager(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        subfolder: str = "",
        tokenizer_huggingface_repo_id: str | None = None,
        tokenizer_subfolder: str | None = None,
    ):
        if tokenizer_huggingface_repo_id is None:
            tokenizer_huggingface_repo_id = huggingface_repo_id
        if tokenizer_subfolder is None:
            tokenizer_subfolder = subfolder
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_huggingface_repo_id, subfolder=tokenizer_subfolder
        )

        reference_dataset = import_encoder_dataset_from_hugging_face(
            huggingface_repo_id,
            subfolder=subfolder,
            tokenizer_config=get_tokenizer_config(
                tokenizer_huggingface_repo_id, subfolder=tokenizer_subfolder
            ),
        )
        target_dataset = copy(reference_dataset)

        reference_dataset.root_theta = reference_dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=reference_dtype)
        )
        config = T5Config.from_properties(
            reference_dataset.properties,
        )

        target_dataset.root_theta = target_dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )

        target_model = T5Encoder(theta=target_dataset.root_theta, config=config)
        reference_model = T5Encoder(theta=reference_dataset.root_theta, config=config)

        input_ids = tokenizer(
            get_test_prompts(),
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        input_kwargs = OrderedDict([("input_ids", input_ids)])
        batch_size = input_ids.shape[0]

        self.compare_iree_vs_eager(
            target_model=target_model,
            reference_model=reference_model,
            input_kwargs=input_kwargs,
            assert_close=assert_t5_encoder_state_close,
        )

    @parameterized.expand(
        [
            (torch.float32, torch.float64, 1e-5, 1e-5),
            (
                torch.bfloat16,
                torch.float64,
                1e-1,
                1e-2,
            ),
        ]
    )
    def testCompareToyIreeVsEager(
        self,
        target_dtype: torch.dtype,
        reference_dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        config = get_t5_encoder_toy_config()
        target_theta = make_t5_encoder_random_theta(config, dtype=target_dtype)
        target_model = T5Encoder(theta=target_theta, config=config)
        _, input_kwargs = target_model.sample_inputs(batch_size=2)

        def assert_close(actual: torch.Tensor, expected: torch.Tensor):
            torch.testing.assert_close(
                actual.to(dtype=expected.dtype), expected, atol=atol, rtol=rtol
            )

        self.compare_iree_vs_eager(
            target_model=target_model,
            input_kwargs=input_kwargs,
            assert_close=assert_close,
            reference_dtype=reference_dtype,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallIreeF32AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1SmallIreeBf16AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlIreeF32AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlIreeBf16AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    @pytest.mark.expensive
    def testCompareV1_1XxlFluxRepoIreeBf16AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "black-forest-labs/FLUX.1-dev",
            subfolder="text_encoder_2",
            tokenizer_subfolder="tokenizer_2",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
        )


class T5AttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            vocab_size=11,
            d_model=13,
            d_kv=7,
            d_ff=3,
            num_heads=2,
            relative_attention_num_buckets=5,
            relative_attention_max_distance=17,
            dropout_rate=0.0,
        )
        reference_model = ReferenceT5Attention(
            reference_config, has_relative_attention_bias=True
        )
        reference_model.eval()

        theta = Theta(
            {
                "q.weight": DefaultPrimitiveTensor(
                    data=reference_model.q.weight.to(dtype=target_dtype)
                ),
                "k.weight": DefaultPrimitiveTensor(
                    data=reference_model.k.weight.to(dtype=target_dtype)
                ),
                "v.weight": DefaultPrimitiveTensor(
                    data=reference_model.v.weight.to(dtype=target_dtype)
                ),
                "o.weight": DefaultPrimitiveTensor(
                    data=reference_model.o.weight.to(dtype=target_dtype)
                ),
                "relative_attention_bias.weight": DefaultPrimitiveTensor(
                    data=reference_model.relative_attention_bias.weight.to(
                        dtype=target_dtype
                    )
                ),
            }
        )
        model = T5Attention(
            theta=theta,
            is_decoder=reference_config.is_decoder,
            relative_attention_num_buckets=reference_config.relative_attention_num_buckets,
            relative_attention_max_distance=reference_config.relative_attention_max_distance,
            d_model=reference_config.d_model,
            d_kv=reference_config.d_kv,
            num_heads=reference_config.num_heads,
            activation_dtype=target_dtype,
            has_relative_attention_bias=True,
        )
        model.eval()

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        reference_mask = make_random_mask(
            shape=[batch_size, 1, 1, batch_seq_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            mask=reference_mask,
            query_length=batch_seq_len,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        mask = ops.to(reference_mask, dtype=target_dtype)
        actual_outputs = model(
            hidden_states=hidden_states,
            mask=mask,
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareSelfAttentionAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            vocab_size=11,
            d_model=13,
            d_kv=7,
            d_ff=3,
            num_heads=2,
            relative_attention_num_buckets=5,
            relative_attention_max_distance=17,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
        )
        reference_model = ReferenceT5LayerSelfAttention(
            reference_config, has_relative_attention_bias=True
        )
        reference_model.eval()

        theta = Theta(
            {
                "SelfAttention.q.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.q.weight.to(dtype=target_dtype)
                ),
                "SelfAttention.k.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.k.weight.to(dtype=target_dtype)
                ),
                "SelfAttention.v.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.v.weight.to(dtype=target_dtype)
                ),
                "SelfAttention.o.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.o.weight.to(dtype=target_dtype)
                ),
                "SelfAttention.relative_attention_bias.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.relative_attention_bias.weight.to(
                        dtype=target_dtype
                    )
                ),
                "layer_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight.to(dtype=target_dtype)
                ),
            }
        )
        model = T5SelfAttention(
            theta=theta,
            is_decoder=reference_config.is_decoder,
            relative_attention_num_buckets=reference_config.relative_attention_num_buckets,
            relative_attention_max_distance=reference_config.relative_attention_max_distance,
            d_model=reference_config.d_model,
            d_kv=reference_config.d_kv,
            num_heads=reference_config.num_heads,
            activation_dtype=torch.float32,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            has_relative_attention_bias=True,
        )
        model.eval()

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        reference_mask = make_random_mask(
            shape=[batch_size, 1, 1, batch_seq_len], dtype=reference_dtype
        )
        reference_position_bias = make_rand_torch(
            shape=[
                batch_size,
                reference_config.num_heads,
                batch_seq_len,
                batch_seq_len,
            ],
            dtype=reference_dtype,
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_mask,
            position_bias=reference_position_bias,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        mask = ops.to(reference_mask, dtype=target_dtype)
        position_bias = ops.to(reference_position_bias, dtype=target_dtype)
        actual_outputs = model(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_bias=position_bias,
        )
        actual_outputs = [
            unbox_tensor(t) if t is not None else t for t in actual_outputs
        ]
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)


class T5LayerFFTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            d_model=13,
            d_ff=3,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="gated-gelu",
        )

        reference_model = ReferenceT5LayerFF(reference_config)
        reference_model.eval()

        theta = Theta(
            {
                "DenseReluDense.wi_0.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wi_0.weight.to(
                        dtype=target_dtype
                    )
                ),
                "DenseReluDense.wi_1.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wi_1.weight.to(
                        dtype=target_dtype
                    )
                ),
                "DenseReluDense.wo.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wo.weight.to(dtype=target_dtype)
                ),
                "layer_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight.to(dtype=target_dtype)
                ),
            }
        )
        model = T5LayerFF(
            theta=theta,
            is_gated_act=reference_config.is_gated_act,
            dense_act_fn=reference_config.dense_act_fn,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            activation_dtype=torch.float32,
        )

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        expected_output = reference_model(
            hidden_states=reference_hidden_states,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        actual_output = model(
            hidden_states=hidden_states,
        )
        actual_output = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_output,
        )

        assert_tensor_close(actual_output, expected_output, atol=atol, rtol=rtol)
