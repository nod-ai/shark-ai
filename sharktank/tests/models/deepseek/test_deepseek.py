# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import pytest
import unittest
import iree
from copy import deepcopy

import torch

from sharktank.models.llm import *
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.utils.create_cache import create_paged_kv_cache
from sharktank.utils.export_artifacts import ExportArtifacts, IreeBenchmarkException
from sharktank.utils.iree import (
    TorchLikeIreeModule,
    get_iree_devices,
    load_iree_module,
    make_hal_buffer_view_trace_default_callback,
    with_iree_device_context,
)
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *
from sharktank.utils.testing import is_mi300x, TempDirTestBase
from sharktank.utils import debugging
import os


@pytest.mark.usefixtures("get_iree_flags")
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

    @is_mi300x
    def testUnshardedToySizedModelIREEVsEager(self):
        def tranfer_to_device(node: Theta | dict | AnyTensor, dev: torch.device):
            if isinstance(node, AnyTensor):
                return node.to(dev)

            if isinstance(node, dict):
                for key in node.keys():
                    node[key] = tranfer_to_device(node[key], dev)
                return node

            for key in node.keys:
                tranfer_to_device(node.tensor(key), dev)

        # 0. Setup
        work_dir = self._temp_dir
        theta, config = generate(12345)
        config.device = torch.device("cuda:0")

        tranfer_to_device(theta, config.device)

        eager_logits_path = work_dir / "reference_logits.npy"
        iree_logits_name = "iree_logits.npy"
        iree_logits_path = work_dir / iree_logits_name
        token_ids_path = work_dir / "token_ids.npy"
        seq_lens_path = work_dir / "seq_lens.npy"
        seq_block_ids_path = work_dir / "seq_block_ids_before_prefill.npy"
        iree_cache_state_paths = [
            work_dir / f"iree_cache_state_{i}.npy"
            for i in range(
                config.pipeline_parallelism_size * config.tensor_parallelism_size
            )
        ]

        dataset_path = work_dir / "parameters.irpa"
        output_name = work_dir / "toy_deepseek"

        # 1. Get inputs
        ids = [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
            [3, 5, 2, 1],
        ]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids, device=config.device)
        np.save(token_ids_path, token_ids.cpu().numpy())
        token_ids = torch.tensor(np.load(token_ids_path), device=config.device)

        batch_size = token_ids.shape[0]

        seq_lens = torch.as_tensor(seq_lens, device=config.device)
        np.save(seq_lens_path, seq_lens.cpu().numpy())
        seq_lens = torch.tensor(np.load(seq_lens_path), device=config.device)

        Dataset(root_theta=theta, properties=config.to_properties()).save(
            path=dataset_path
        )

        eager_model = PagedLlmModelV1(theta=theta, config=config)
        generator = TorchGenerator(eager_model)
        eager_batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )

        # TODO: How come only the iree_cache gets sharded?
        cache_state_before_prefill = deepcopy(eager_batch.cache_state)
        iree_cache = create_paged_kv_cache(config)
        iree_cache_state = iree_cache.shard_state(deepcopy(cache_state_before_prefill))
        for i in range(len(iree_cache_state)):
            state_i = iree_cache_state[i]
            np.save(iree_cache_state_paths[i], state_i.cpu().numpy())
            iree_cache_state[i] = torch.tensor(
                np.load(iree_cache_state_paths[i]), device=token_ids.device
            )

        seq_block_ids_before_prefill = eager_batch.pad_block_ids()
        np.save(
            seq_block_ids_path,
            seq_block_ids_before_prefill.cpu().numpy(),
        )

        prefill_args = [
            f"--function=prefill_bs{batch_size}",
            f"--input=@{token_ids_path}",
            f"--input=@{seq_lens_path}",
            f"--input=@{seq_block_ids_path}",
            *(f"--input=@{path}" for path in iree_cache_state_paths),
        ]

        # 2. Run Eager
        eager_batch.prefill()
        eager_logits = eager_batch.prefill_logits
        np.save(eager_logits_path, eager_logits.cpu().numpy())
        eager_logits = torch.tensor(np.load(eager_logits_path), device=token_ids.device)

        # 3. Run IREE
        config.device = torch.device("cpu")  # Switch back to cpu for tracing
        exporter = ExportArtifacts.from_config(
            config,
            irpa_path=dataset_path,
            batch_size=batch_size,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            hip_device_id=self.iree_device,
            output_name=output_name,
        )
        exporter.export_and_compile_llm(skip_decode=True)
        iree_logits = exporter.iree_run(
            run_args=prefill_args, output_paths=[iree_logits_path]
        )

        # 4. Compare outputs
        for i, (eager_logits_i, iree_logits_i) in enumerate(
            zip(eager_logits, iree_logits)
        ):
            assert eager_logits_i.shape == iree_logits_i.shape
            same = torch.isclose(eager_logits_i, iree_logits_i, rtol=1.3e-6, atol=1e-5)
            if not same.all():
                raise AssertionError(
                    f"Logits mismatch for batch {i}: "
                    f"Num mismatch: {(~same).sum()}. {100*same.sum() / same.numel():.1f}% match."
                )
