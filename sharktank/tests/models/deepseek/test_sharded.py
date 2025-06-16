# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest
from copy import deepcopy

import torch
import iree
import pytest
import torch
import unittest
from parameterized import parameterized

from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.models.llm import PagedLlmModelV1
from sharktank.types.pipelining import pipeline_parallelize_theta
from sharktank.types import Dataset
from sharktank.types.sharding import shard_theta
from sharktank.utils.evaluate import pad_tokens
from sharktank.utils.export_artifacts import ExportArtifacts, IreeCompileException
from sharktank.utils.load_llm import TorchGenerator
from sharktank.utils.create_cache import *
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    TorchLikeIreeModule,
    with_iree_device_context,
)
from sharktank.utils.testing import TempDirTestBase
from sharktank.examples.sharding import shard_llm_dataset


@pytest.mark.usefixtures("get_iree_flags")
class DeepseekShardedTest(TempDirTestBase):
    @parameterized.expand(
        [
            (2, 1),
            (1, 2),
            (2, 2),
        ]
    )
    def testParallelToySizedModelEagerVsUnsharded(
        self, tensor_parallelism_size: int, pipeline_parallelism_size: int
    ):
        theta, config = generate(12345)

        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = tensor_parallelism_size
        sharded_config.pipeline_parallelism_size = pipeline_parallelism_size
        if tensor_parallelism_size > 1:
            sharded_theta = shard_theta(theta=theta, config=sharded_config)
        else:
            sharded_theta = deepcopy(theta)

        block_to_pipeline, pipeline_to_devices = pipeline_parallelize_theta(
            sharded_theta, pipeline_parallelism_size
        )
        sharded_config.block_to_pipeline_map = block_to_pipeline
        sharded_config.pipeline_to_device_map = pipeline_to_devices

        reference_model = PagedLlmModelV1(theta=theta, config=config)
        target_model = PagedLlmModelV1(theta=sharded_theta, config=sharded_config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        reference_generator = TorchGenerator(reference_model)
        reference_batch = reference_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        reference_batch.prefill()
        reference_logits = reference_batch.prefill_logits

        target_generator = TorchGenerator(target_model)
        target_batch = target_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        target_batch.prefill()
        target_logits = target_batch.prefill_logits

        torch.testing.assert_close(
            target_logits, reference_logits, atol=2e-4, rtol=2e-2
        )

        # TODO: test decode step and maybe verify the paged cache is close.

    @pytest.mark.xfail(
        reason="https://github.com/nod-ai/shark-ai/issues/1566",
        strict=True,
    )
    def testTensorParallelToySizedModelIREEVsUnshardedEager(self):
        theta, config = generate(12345)
        tensor_parallelism_size = 2

        ids = [
            [1, 2, 3, 4],
            [10, 9, 8, 7, 6, 5],
        ]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)
        batch_size = token_ids.shape[0]

        dataset = Dataset(root_theta=theta, properties=config.to_properties())
        dataset_path = self._temp_dir / "parameters.irpa"
        dataset.save(path=dataset_path)

        sharded_parameters_path = self._temp_dir / "parameters.irpa"
        shard_llm_dataset.main(
            [
                f"--tensor-parallelism-size={tensor_parallelism_size}",
                f"--irpa-file={dataset_path}",
                f"--output-irpa-file={sharded_parameters_path}",
            ]
        )
        sharded_dataset = Dataset.load(sharded_parameters_path)
        sharded_config = LlamaModelConfig.from_properties(sharded_dataset.properties)

        reference_model = PagedLlmModelV1(theta=theta, config=config)
        reference_generator = TorchGenerator(reference_model)
        reference_batch = reference_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        cache_state_before_prefill = deepcopy(reference_batch.cache_state)
        seq_block_ids_before_prefill = reference_batch.pad_block_ids()
        reference_batch.prefill()
        reference_logits = reference_batch.prefill_logits

        sharded_cache = create_paged_kv_cache(sharded_config)
        sharded_cache_state = sharded_cache.shard_state(
            deepcopy(cache_state_before_prefill)
        )

        mlir_path = self._temp_dir / "model.mlir"
        export_config_path = self._temp_dir / "model_export_config.json"
        export_artifacts = ExportArtifacts.from_config(
            sharded_config,
            irpa_path=str(sharded_parameters_path),
            batch_size=batch_size,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            iree_hal_local_target_device_backends=self.iree_hal_local_target_device_backends,
        )
        export_artifacts.export_to_mlir(
            output_mlir=str(mlir_path),
            output_config=str(export_config_path),
            skip_decode=True,  # TODO: enable decode
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        export_artifacts.compile_to_vmfb(
            output_mlir=str(mlir_path),
            output_vmfb=str(iree_module_path),
            args=[],
        )

        iree_devices = get_iree_devices(
            device=self.iree_device,
            device_count=tensor_parallelism_size,
        )

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):

            iree_module, vm_context, vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=sharded_parameters_path,
                tensor_parallel_size=tensor_parallelism_size,
            )

            torch_like_iree_module = TorchLikeIreeModule(
                module=iree_module, devices=iree_devices, vm_context=vm_context
            )
            args = (
                token_ids,
                seq_lens,
                seq_block_ids_before_prefill,
                sharded_cache_state,
            )
            iree_result = getattr(torch_like_iree_module, f"prefill_bs{batch_size}")(
                *args
            )

            # Make sure we don't leak IREE-backed tensors outside of this function.
            iree_result = [t.clone() for t in iree_result]
            iree_logits = iree_result[0]
            return iree_logits

        iree_logits = with_iree_device_context(run_iree_module, iree_devices)
        torch.testing.assert_close(iree_logits, reference_logits)
