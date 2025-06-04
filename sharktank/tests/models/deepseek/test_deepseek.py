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
from sharktank.utils.testing import TempDirTestBase
from sharktank.utils import debugging


# @pytest.mark.usefixtures("get_iree_flags")
class DeepseekTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        self.callback_stash = debugging.get_trace_tensor_callback()
        debugging.set_trace_tensor_callback(
            debugging.trace_tensor_to_safetensors_callback
        )

        self.enable_tensor_trace_stash = debugging.flags.enable_tensor_trace
        # debugging.flags.enable_tensor_trace = True

        self.trace_path_stash = debugging.flags.trace_path
        debugging.flags.trace_path = Path(
            "/home/alvasile/repos/shark-ai/sharktank/logits"
        )

    def tearDown(self):
        super().tearDown()
        debugging.set_trace_tensor_callback(self.callback_stash)
        debugging.flags.enable_tensor_trace = self.enable_tensor_trace_stash
        debugging.flags.trace_path = self.trace_path_stash

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
        work_dir = self._temp_dir
        theta, config = generate(12345)

        ids = [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
            [3, 5, 2, 1],
        ]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)
        batch_size = token_ids.shape[0]

        dataset = Dataset(root_theta=theta, properties=config.to_properties())
        dataset_path = work_dir / "parameters.irpa"
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

        mlir_path = work_dir / "model.mlir"
        export_config_path = work_dir / "model_export_config.json"
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

        iree_module_path = work_dir / "model.vmfb"
        export_artifacts.compile_to_vmfb(
            output_mlir=mlir_path,
            output_vmfb=iree_module_path,
            args=[],
        )

        iree_devices = get_iree_devices(
            device="hip://4",
            device_count=1,
        )

        token_ids_path = work_dir / "token_ids.npy"
        seq_lens_path = work_dir / "seq_lens.npy"
        seq_block_ids_before_prefill_path = (
            work_dir / "seq_block_ids_before_prefill.npy"
        )
        iree_cache_state_path = work_dir / "iree_cache_state.npy"

        np.save(token_ids_path, token_ids.cpu().numpy())
        np.save(seq_lens_path, seq_lens.cpu().numpy())
        np.save(
            seq_block_ids_before_prefill_path,
            seq_block_ids_before_prefill.cpu().numpy(),
        )
        np.save(iree_cache_state_path, iree_cache_state[0].cpu().numpy())

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            cpu_device = get_iree_devices(driver="local-task", device_count=1)
            iree_buffere_view_trace_callback = (
                make_hal_buffer_view_trace_default_callback(cpu_device[0])
            )
            debug_sink = iree.runtime.HalModuleDebugSink(
                iree_buffere_view_trace_callback
            )

            iree_module, vm_context, vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=dataset_path,
                debug_sink=debug_sink,
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

        iree_logits_w_py = with_iree_device_context(run_iree_module, iree_devices)
        iree_cache_state_after = deepcopy(iree_cache_state)

        run_args = [
            "iree-run-module",
            "--hip_use_streams=true",
            f"--parameters=model={dataset_path}",
            f"--module={iree_module_path}",
            "--device=hip://0",
            f"--function=prefill_bs{batch_size}",
            f"--input=@{token_ids_path}",
            f"--input=@{seq_lens_path}",
            f"--input=@{seq_block_ids_before_prefill_path}",
            f"--input=@{iree_cache_state_path}",
            "--output=@iree_logits.npy",
        ]
        cmd = subprocess.list2cmdline(run_args)
        print(f" Launching compile command:\n" f"cd {work_dir} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=work_dir)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeBenchmarkException(proc, work_dir)

        iree_logits_cli = torch.tensor(np.load(work_dir / "iree_logits.npy"))

        # Compare logits
        padding_mask = (
            (token_ids != 0).int().detach().clone().to(token_ids.device).bool()
        )
        all_ref_logits, all_iree_logits_w_py, all_iree_logits = [], [], []
        for i in range(len(ids)):
            all_ref_logits.append(reference_logits[i, padding_mask[i]])
            all_iree_logits_w_py.append(iree_logits_w_py[i, padding_mask[i]])
            all_iree_logits.append(iree_logits_cli[i, padding_mask[i]])

        for i, (iree_logits_py_i, iree_logits_cli_i) in enumerate(
            zip(all_iree_logits_w_py, all_iree_logits)
        ):
            assert iree_logits_py_i.shape == iree_logits_cli_i.shape
            same = torch.isclose(
                iree_logits_py_i, iree_logits_cli_i, rtol=1.3e-6, atol=1e-5
            )
            if not same.all():
                raise AssertionError(
                    f"Logits mismatch for batch {i}: "
                    f"Num mismatch: {(~same).sum()}. {100*same.sum() / same.numel():.1f}% match."
                )

        for i, (ref_logits_i, iree_logits_py_i) in enumerate(
            zip(all_ref_logits, all_iree_logits_w_py)
        ):
            assert ref_logits_i.shape == iree_logits_py_i.shape
            same = torch.isclose(ref_logits_i, iree_logits_py_i, rtol=1.3e-6, atol=1e-5)
            if not same.all():
                raise AssertionError(
                    f"Logits mismatch for batch {i}: "
                    f"Num mismatch: {(~same).sum()}. {100*same.sum() / same.numel():.1f}% match."
                )

        # Compare cache state
        assert len(iree_cache_state_after) == len(reference_cache_state_after)
        for i in range(len(iree_cache_state_after)):
            iree_state_i = iree_cache_state_after[i]
            reference_state_i = reference_cache_state_after[i]
            torch.testing.assert_close(iree_state_i, reference_state_i)
