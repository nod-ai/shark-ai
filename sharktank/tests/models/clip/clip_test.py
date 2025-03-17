# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict
import functools
import iree.compiler
import iree.runtime
import os
from pathlib import Path
from parameterized import parameterized
from copy import copy
import logging
import pytest
import torch
from torch.utils._pytree import tree_map
from typing import Optional
from unittest import TestCase
import transformers
from transformers import CLIPTextModel as HfCLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import (
    CLIPAttention as HfCLIPAttention,
    CLIPEncoderLayer as HfCLIPEncoderLayer,
    CLIPEncoder as HfCLIPEncoder,
)

from sharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from sharktank.types import (
    DefaultPrimitiveTensor,
    dtype_to_serialized_short_name,
    Dataset,
)
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils.hf_datasets import get_dataset
from sharktank.utils.testing import (
    assert_text_encoder_state_close,
    make_rand_torch,
    make_random_mask,
    TempDirTestBase,
    test_prompts,
)
from sharktank.models.clip.export import (
    export_clip_text_model_dataset_from_hugging_face,
    hugging_face_clip_attention_to_theta,
    hugging_face_clip_encoder_layer_to_theta,
    hugging_face_clip_encoder_to_theta,
    hugging_face_clip_text_model_to_dataset,
    hugging_face_clip_text_model_to_theta,
)
from sharktank.models.clip.testing import (
    make_random_input_token_sequences,
    make_clip_text_model_random_theta,
    export_clip_text_model_iree_test_data,
    clip_toy_text_model_config,
)
from sharktank.models.clip import (
    ClipAttention,
    ClipEncoderLayer,
    ClipEncoder,
    ClipTextModel,
)
from sharktank.layers.configs.llm_configs import ClipTextConfig
from sharktank import ops

with_clip_data = pytest.mark.skipif("not config.getoption('with_clip_data')")

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("path_prefix")
class ClipTextIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        if self.path_prefix is None:
            self.path_prefix = self._temp_dir
        else:
            self.path_prefix = Path(self.path_prefix)

    @with_clip_data
    def testSmokeExportLargeF32FromHuggingFace(self):
        huggingface_repo_id = "openai/clip-vit-large-patch14"
        huggingface_repo_id_as_path = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}"
        )
        get_dataset(
            huggingface_repo_id,
        ).download()
        target_dtype_name = dtype_to_serialized_short_name(torch.float32)
        target_model_path_prefix = (
            self.path_prefix
            / f"{huggingface_repo_id_as_path}_text_model_{target_dtype_name}"
        )
        output_path = f"{target_model_path_prefix}.irpa"
        export_clip_text_model_dataset_from_hugging_face(
            huggingface_repo_id, output_path
        )

    def testSmokeExportToyIreeTestData(self):
        from sharktank.models.clip.export_toy_text_model_iree_test_data import main

        main([f"--output-dir={self.path_prefix/'clip_toy_text_model'}"])

    @with_clip_data
    def testCompareLargeIreeF32AgainstTorchEagerF32(self):
        self.runTestCompareIreeAgainstPretrainedTorchEager(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-5,
        )

    @with_clip_data
    def testCompareLargeIreeBf16AgainstTorchEagerF32(self):
        self.runTestCompareIreeAgainstPretrainedTorchEager(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            # The observed error is 1.43e-2. We leave a bit of margin.
            atol=3e-3,
        )

    @with_clip_data
    def testCompareToyModelIreeF32AgainstTorchEagerF32(self):
        self.runTestCompareToyModelIreeAgainstTorch(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-5
        )

    @with_clip_data
    def testCompareToyModelIreeBf16AgainstTorchEagerF32(self):
        self.runTestCompareToyModelIreeAgainstTorch(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16, atol=1e-3
        )

    @torch.no_grad()
    def runTestCompareIreeAgainstTorchEagerWithInputTokens(
        self,
        reference_model: ClipTextModel,
        target_dtype: torch.dtype,
        input_ids: torch.LongTensor,
        atol: float,
        file_artifact_prefix_name: str,
    ):
        reference_dtype_name = dtype_to_serialized_short_name(
            reference_model.config.dtype
        )
        target_dtype_name = dtype_to_serialized_short_name(target_dtype)
        reference_model_path_prefix = (
            self.path_prefix / f"{file_artifact_prefix_name}_{reference_dtype_name}"
        )
        target_model_path_prefix = (
            self.path_prefix / f"{file_artifact_prefix_name}_{target_dtype_name}"
        )

        parameters_path = f"{target_model_path_prefix}.irpa"
        input_args = OrderedDict([("input_ids", input_ids)])
        batch_size = input_ids.shape[0]
        mlir_path = f"{target_model_path_prefix}.mlir"

        logger.info("Exporting clip text model to MLIR...")
        export_clip_text_model_iree_test_data(
            reference_model=reference_model,
            target_dtype=target_dtype,
            input_ids=input_ids,
            target_mlir_output_path=mlir_path,
            target_iree_parameters_output_path=parameters_path,
        )

        iree_module_path = f"{target_model_path_prefix}.vmfb"
        logger.info("Compiling MLIR file...")
        iree.compiler.compile_file(
            mlir_path,
            output_file=iree_module_path,
            extra_args=["--iree-hal-target-device=hip", "--iree-hip-target=gfx942"],
        )

        logger.info("Invoking reference torch function...")
        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            kwargs=input_args,
            trace_path_prefix=f"{reference_model_path_prefix}_torch_",
        )
        expected_outputs = flatten_for_iree_signature(reference_result_dict)

        iree_devices = get_iree_devices(driver="hip", device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            logger.info("Loading IREE module...")
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=parameters_path,
            )
            iree_args = prepare_iree_module_function_args(
                args=flatten_for_iree_signature(input_args), devices=iree_devices
            )
            logger.info("Invoking IREE function...")
            iree_result = iree_to_torch(
                *run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name=f"forward_bs{batch_size}",
                    trace_path_prefix=f"{target_model_path_prefix}_iree_",
                )
            )
            actual_outputs = [
                ops.to(iree_result[i], dtype=expected_outputs[i].dtype)
                for i in range(len(expected_outputs))
            ]
            return [t.clone() for t in actual_outputs]

        actual_outputs = with_iree_device_context(run_iree_module, iree_devices)

        actual_last_hidden_state = actual_outputs[0]
        expected_last_hidden_state = expected_outputs[0]

        logger.info("Comparing outputs...")
        assert_text_encoder_state_close(
            actual_last_hidden_state, expected_last_hidden_state, atol
        )

    def runTestCompareRandomModelIreeAgainstTorch(
        self,
        reference_config: ClipTextConfig,
        target_dtype: torch.dtype,
        batch_size: int,
        atol: float,
        file_artifact_prefix_name: str,
    ):
        input_ids = make_random_input_token_sequences(
            batch_size=batch_size, config=reference_config
        )
        reference_theta = make_clip_text_model_random_theta(reference_config)
        reference_model = ClipTextModel(theta=reference_theta, config=reference_config)
        self.runTestCompareIreeAgainstTorchEagerWithInputTokens(
            reference_model=reference_model,
            target_dtype=target_dtype,
            input_ids=input_ids,
            atol=atol,
            file_artifact_prefix_name=file_artifact_prefix_name,
        )

    def runTestCompareToyModelIreeAgainstTorch(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        batch_size = 4
        reference_config = clip_toy_text_model_config(reference_dtype)
        file_artifact_prefix_name = "clip_text_model_toy"
        self.runTestCompareRandomModelIreeAgainstTorch(
            reference_config=reference_config,
            target_dtype=target_dtype,
            batch_size=batch_size,
            atol=atol,
            file_artifact_prefix_name=file_artifact_prefix_name,
        )

    def runTestCompareIreeAgainstPretrainedTorchEager(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()

        huggingface_repo_id_as_path = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}"
        )
        file_artifact_prefix_name = f"{huggingface_repo_id_as_path}_text_model"

        hf_model: HfCLIPTextModel = HfCLIPTextModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )
        reference_dataset = hugging_face_clip_text_model_to_dataset(hf_model)
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            hf_model.config
        )
        reference_model = ClipTextModel(
            theta=reference_dataset.root_theta, config=config
        )

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(huggingface_repo_id)
        input_ids = tokenizer(
            test_prompts,
            truncation=True,
            max_length=reference_model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        self.runTestCompareIreeAgainstTorchEagerWithInputTokens(
            reference_model=reference_model,
            target_dtype=target_dtype,
            input_ids=input_ids,
            atol=atol,
            file_artifact_prefix_name=file_artifact_prefix_name,
        )


@pytest.mark.usefixtures("get_model_artifacts")
class ClipTextEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    def runTestCompareTorchEagerAgainstHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: float,
    ):
        """Compares the last hidden states with the cosine similarity metric.
        This metric is sensible as the outputs are the result of layer normalization.
        The angle between the vectors would indicate how close they are."""
        get_dataset(
            huggingface_repo_id,
        ).download()

        reference_model: HfCLIPTextModel = HfCLIPTextModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )

        theta = hugging_face_clip_text_model_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_model.config
        )
        model = ClipTextModel(theta, config)

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            huggingface_repo_id,
            max_length=reference_model.config.max_position_embeddings,
        )
        input_ids = tokenizer(
            test_prompts,
            truncation=True,
            max_length=reference_model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        expected_outputs = reference_model(input_ids=input_ids)
        actual_outputs = model(input_ids=DefaultPrimitiveTensor(data=input_ids))
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_text_encoder_state_close(
            actual_outputs["last_hidden_state"],
            expected_outputs["last_hidden_state"],
            atol=atol,
        )

    @with_clip_data
    def testLargeCompareTorchEagerF32AgainstHuggingFaceF32(self):
        self.runTestCompareTorchEagerAgainstHuggingFace(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-5,
        )

    @with_clip_data
    def testLargeCompareTorchEagerBf16AgainstHuggingFaceF32(self):
        self.runTestCompareTorchEagerAgainstHuggingFace(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            # The observed error is 1.5e-3. We leave a bit of margin.
            atol=3e-3,
        )

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 4e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 4e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        vocab_size = 11
        config = clip_toy_text_model_config()
        reference_config = config.to_hugging_face_clip_text_model_config()
        reference_model = HfCLIPTextModel(
            reference_config,
        )
        reference_model.eval()

        theta = hugging_face_clip_text_model_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_config
        )
        model = ClipTextModel(theta, config)

        input_ids = torch.randint(
            low=0, high=vocab_size, size=[batch_size, config.max_position_embeddings]
        )

        expected_outputs = reference_model(input_ids=input_ids)

        actual_outputs = model(input_ids=DefaultPrimitiveTensor(data=input_ids))
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            # Default values are not enough because torch.nn.Linear does fused
            # multiply-add, while our implementation is decomposed.
            # There may be other source of discrepancy.
            [torch.bfloat16, torch.bfloat16, 0.5e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        config = clip_toy_text_model_config()
        reference_config = config.to_hugging_face_clip_text_model_config()
        tgt_len = config.max_position_embeddings
        src_len = tgt_len
        reference_model = HfCLIPAttention(
            reference_config,
        )
        reference_model.eval()

        theta = hugging_face_clip_attention_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_config
        )
        model = ClipAttention(theta, config)

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, tgt_len, config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipEncoderLayerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 1e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        config = clip_toy_text_model_config()
        reference_config = config.to_hugging_face_clip_text_model_config()
        tgt_len = config.max_position_embeddings
        src_len = tgt_len
        reference_model = HfCLIPEncoderLayer(
            reference_config,
        )
        reference_model.eval()

        theta = hugging_face_clip_encoder_layer_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_config
        )
        model = ClipEncoderLayer(theta, config)

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, tgt_len, reference_config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipEncoderTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 2e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 2e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        tgt_len = 23
        src_len = tgt_len
        reference_config = (
            clip_toy_text_model_config().to_hugging_face_clip_text_model_config()
        )
        reference_model = HfCLIPEncoder(
            reference_config,
        )
        reference_model.eval()

        theta = hugging_face_clip_encoder_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_config
        )
        model = ClipEncoder(theta, config)

        reference_inputs_embeds = make_rand_torch(
            shape=[batch_size, tgt_len, reference_config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            inputs_embeds=reference_inputs_embeds,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        inputs_embeds = ops.to(reference_inputs_embeds, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            inputs_embeds=DefaultPrimitiveTensor(data=inputs_embeds),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )
