# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict
import functools
import iree.compiler
import iree.runtime
import json
from pathlib import Path
from parameterized import parameterized
import pytest
import torch
from torch.utils._pytree import tree_map
from typing import Optional
from unittest import TestCase
from transformers import CLIPTextModel as HfCLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import (
    CLIPAttention as HfCLIPAttention,
    CLIPEncoderLayer as HfCLIPEncoderLayer,
    CLIPEncoder as HfCLIPEncoder,
)

import iree.runtime
from amdsharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from amdsharktank.types import (
    DefaultPrimitiveTensor,
    dtype_to_serialized_short_name,
    unbox_tensor,
)
from amdsharktank.transforms.dataset import set_float_dtype
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.hf_datasets import get_dataset
from amdsharktank.utils.random import make_random_mask, make_rand_torch
from amdsharktank.utils.testing import (
    is_cpu_condition,
    assert_tensor_close,
    assert_text_encoder_state_close,
    TempDirTestBase,
    get_test_prompts,
)
from amdsharktank.models.clip.export import (
    hugging_face_clip_attention_to_theta,
    hugging_face_clip_encoder_layer_to_theta,
    hugging_face_clip_encoder_to_theta,
    hugging_face_clip_text_model_to_theta,
)
from amdsharktank.models.clip.testing import (
    clip_text_model_from_reference_model,
    clip_toy_text_model_config,
)
from amdsharktank.models.clip import (
    ClipAttention,
    ClipEncoderLayer,
    ClipEncoder,
    ClipTextModel,
)
from amdsharktank.layers.configs.llm_configs import ClipTextConfig
from amdsharktank.layers import (
    ExportFunctionConfig,
    ModelConfig,
    get_model_type_id,
    create_model,
)
from amdsharktank import ops

with_clip_data = pytest.mark.skipif("not config.getoption('with_clip_data')")

logger = get_logger(__name__)


@pytest.mark.usefixtures("path_prefix", "iree_flags")
class ClipTextIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        if self.path_prefix is None:
            self.path_prefix = self._temp_dir
        else:
            self.path_prefix = Path(self.path_prefix)

    @with_clip_data
    @pytest.mark.expensive
    def testSmokeExportLargeF32FromHuggingFace(self):
        parameters_path = self.path_prefix / "model.irpa"
        config_dict = {
            "model_type": get_model_type_id(ClipTextModel),
            "config_version": ModelConfig.current_config_version,
            "clip_config_version": ClipTextConfig.current_clip_config_version,
            "hugging_face_repo_id": "openai/clip-vit-large-patch14",
            "export_parameters_path": str(parameters_path),
        }
        config_path = self.path_prefix / "config.json"
        with open(str(config_path), "w") as f:
            json.dump(config_dict, f)

        model: ClipTextModel = create_model(config_path)
        parameters_path.unlink(missing_ok=True)
        model.export_parameters()
        assert config_path.exists()

    def testSmokeExportToyIreeTestData(self):
        from amdsharktank.models.clip.export_toy_text_model_iree_test_data import main

        main([f"--output-dir={self.path_prefix/'clip_toy_text_model'}"])

    @with_clip_data
    @pytest.mark.expensive
    def testCompareLargeIreeF32AgainstTorchEagerF32(self):
        self.runTestCompareIreeAgainstPretrainedTorchEager(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-5,
        )

    @with_clip_data
    @pytest.mark.expensive
    def testCompareLargeIreeBf16AgainstTorchEagerF32(self):
        self.runTestCompareIreeAgainstPretrainedTorchEager(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            # The observed error is 1.43e-2. We leave a bit of margin.
            atol=3e-3,
        )

    @pytest.mark.xfail(
        is_cpu_condition,
        raises=iree.compiler.CompilerToolError,
        reason="The compiler segfaults https://github.com/iree-org/iree/issues/20283",
    )
    def testCompareToyModelIreeF32AgainstTorchEagerF32(self):
        self.runTestCompareToyModelIreeAgainstTorch(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-5
        )

    @pytest.mark.xfail(
        is_cpu_condition,
        raises=iree.compiler.CompilerToolError,
        reason="The compiler segfaults https://github.com/iree-org/iree/issues/20283",
    )
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
        iree_module_path = f"{target_model_path_prefix}.vmfb"
        export_functions = [
            ExportFunctionConfig(function=None, batch_sizes=[batch_size])
        ]
        target_model = clip_text_model_from_reference_model(
            reference_model=reference_model,
            target_dtype=target_dtype,
            extra_config_kwargs={
                "config_path": None,
                "export_functions": export_functions,
                "mlir_path": mlir_path,
                "export_parameters_path": parameters_path,
                "iree_module_path": iree_module_path,
                "compile_args": [
                    f"--iree-hal-target-device={self.iree_hal_target_device}",
                    f"--iree-hip-target={self.iree_hip_target}",
                ],
            },
        )

        logger.info("Exporting clip text model to MLIR...")
        target_model.export()

        logger.info("Compiling MLIR file...")
        target_model.compile()

        logger.info("Invoking reference torch function...")
        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            kwargs=input_args,
            trace_path_prefix=f"{reference_model_path_prefix}_torch_",
        )
        expected_outputs = flatten_for_iree_signature(reference_result_dict)

        iree_devices = get_iree_devices(device=self.iree_device)

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
            return [unbox_tensor(t).clone() for t in actual_outputs]

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
        reference_model = ClipTextModel(config=reference_config)
        input_ids = reference_model.sample_inputs(batch_size=batch_size,)[
            1
        ]["input_ids"]
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
        reference_theta = hugging_face_clip_text_model_to_theta(hf_model)
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            hf_model.config
        )
        reference_model = ClipTextModel(theta=reference_theta, config=config)

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(huggingface_repo_id)
        input_ids = tokenizer(
            get_test_prompts(),
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


@pytest.mark.usefixtures("model_artifacts")
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
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_model.config
        )
        model = ClipTextModel(theta=theta, config=config)

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            huggingface_repo_id,
            max_length=reference_model.config.max_position_embeddings,
        )
        input_ids = tokenizer(
            get_test_prompts(),
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
    @pytest.mark.expensive
    def testLargeCompareTorchEagerF32AgainstHuggingFaceF32(self):
        self.runTestCompareTorchEagerAgainstHuggingFace(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-5,
        )

    @with_clip_data
    @pytest.mark.expensive
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
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_hugging_face_clip_text_model_config(
            reference_config
        )
        model = ClipTextModel(theta=theta, config=config)

        input_ids = torch.randint(
            low=0, high=vocab_size, size=[batch_size, config.max_position_embeddings]
        )

        expected_outputs = reference_model(input_ids=input_ids)

        actual_outputs = model(input_ids=DefaultPrimitiveTensor(data=input_ids))
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)


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
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)


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
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)


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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        assert_tensor_close(actual_outputs, expected_outputs, atol=atol, rtol=rtol)
