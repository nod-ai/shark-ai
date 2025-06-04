# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import unittest
import iree
from copy import deepcopy

import torch

from sharktank.models.llm import *
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.utils.create_cache import create_paged_kv_cache
from sharktank.utils.export_artifacts import ExportArtifacts
from sharktank.utils.iree import (
    TorchLikeIreeModule,
    get_iree_devices,
    load_iree_module,
    with_iree_device_context,
)
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *
from sharktank.utils.testing import TempDirTestBase


# @pytest.mark.usefixtures("get_iree_flags")
class DeepseekTest(TempDirTestBase):
    def testCrossEntropy(self):
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]

        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        generator = TorchGenerator(model)
        batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )

        batch.prefill()
        logits = batch.prefill_logits

        ids = token_ids[0, :-1]
        logits = logits[0, 1:]
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        assert pytest.approx(9.7477, 1e-4) == cross_entropy

    def testUnshardedToySizedModelIREEVsEager(self):
        theta, config = generate(12345)

        ids = [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
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
        dataset = Dataset.load(dataset_path)

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
        reference_cache_state_after = deepcopy(reference_batch.cache_state)

        iree_cache = create_paged_kv_cache(config)
        iree_cache_state = iree_cache.shard_state(deepcopy(cache_state_before_prefill))

        mlir_path = self._temp_dir / "model.mlir"
        export_config_path = self._temp_dir / "model_export_config.json"
        export_artifacts = ExportArtifacts.from_config(
            config,
            irpa_path=dataset_path,
            batch_size=batch_size,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            # iree_hal_local_target_device_backends=self.iree_hal_local_target_device_backends,
        )
        export_artifacts.export_to_mlir(
            output_mlir=mlir_path,
            output_config=export_config_path,
            skip_decode=True,  # TODO: enable decode
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        export_artifacts.compile_to_vmfb(
            output_mlir=mlir_path,
            output_vmfb=iree_module_path,
            args=[],
        )

        iree_devices = get_iree_devices(
            device="hip://4",
            device_count=1,
        )

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):

            iree_module, vm_context, vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=dataset_path,
            )

            torch_like_iree_module = TorchLikeIreeModule(
                module=iree_module, devices=iree_devices, vm_context=vm_context
            )
            args = (
                token_ids,
                seq_lens,
                seq_block_ids_before_prefill,
                iree_cache_state,
            )
            iree_result = getattr(torch_like_iree_module, f"prefill_bs{batch_size}")(
                *args
            )

            # Make sure we don't leak IREE-backed tensors outside of this function.
            iree_result = [t.clone() for t in iree_result]
            iree_logits = iree_result[0]
            return iree_logits

        iree_logits_i = with_iree_device_context(run_iree_module, iree_devices)
        iree_cache_state_after = deepcopy(iree_cache_state)

        # Compare logits
        padding_mask = (
            (token_ids != 0).int().detach().clone().to(token_ids.device).bool()
        )
        all_ref_logits, all_iree_logits = [], []
        for i in range(len(ids)):
            all_ref_logits.append(reference_logits[i, padding_mask[i]])
            all_iree_logits.append(iree_logits_i[i, padding_mask[i]])

        for ref_logits_i, iree_logits_i in zip(all_ref_logits, all_iree_logits):
            assert ref_logits_i.shape == iree_logits_i.shape
            torch.testing.assert_close(ref_logits_i, iree_logits_i)

        # Compare cache state
        assert len(iree_cache_state_after) == len(reference_cache_state_after)
        for i in range(len(iree_cache_state_after)):
            iree_state_i = iree_cache_state_after[i]
            reference_state_i = reference_cache_state_after[i]
            torch.testing.assert_close(iree_state_i, reference_state_i)
