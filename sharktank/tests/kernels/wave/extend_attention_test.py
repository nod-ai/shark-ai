# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
import pytest
import torch
from iree.compiler.passmanager import PassManager
from iree.compiler.ir import Context, Module
import iree.turbine.aot as aot
from sharktank.kernels.wave.extend_attention import wave_extend_attention
from parameterized import parameterized
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
from sharktank.types.tensors import unbox_tensor
from sharktank.utils.testing import assert_cosine_similarity_close
import iree.compiler as ireec
import iree.runtime as ireert
from pathlib import Path
import numpy as np
from sharktank.utils.testing import is_mi300x, is_mi350x, IreeFlags


# @is_mi300x
@pytest.mark.usefixtures("iree_flags")
class TestExtendAttention:
    def hip_flags(self):
        return [
            "--iree-hip-target={self.iree_hip_target}",
            "--iree-hal-target-device={self.iree_hal_target_device}",
            "--iree-opt-level=O3",
            "--iree-dispatch-creation-propagate-collapse-across-expands=true",
            "--iree-codegen-enable-default-tuning-specs=true",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hip-specialize-dispatches",
            "--iree-hal-memoization=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
            "--iree-dispatch-creation-enable-early-trunc-fusion=true",
        ]

    @pytest.mark.skipif(
        torch.__version__ < (2, 6),
        reason="Wave extend attention kernel requires torch version >= 2.6",
    )
    @pytest.mark.parametrize(
        "query_seq_len, kv_seq_len, s, num_query_heads, head_size, num_kv_heads, head_size_kv, max_len_extend",
        [
            (879, 879, 3, 16, 128, 1, 128, 458),
        ],
    )
    def test_extend_attention_export_compile_run(
        self,
        iree_flags: IreeFlags,
        tmp_path: Path,
        query_seq_len: int,
        kv_seq_len: int,
        s: int,
        num_query_heads: int,
        head_size: int,
        num_kv_heads: int,
        head_size_kv: int,
        max_len_extend: int,
    ):
        class WaveExtendAttentionModule(torch.nn.Module):
            def forward(
                self,
                q_extend,
                k_extend,
                v_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                max_len_extend_tensor,
            ):
                return wave_extend_attention(
                    q_extend,
                    k_extend,
                    v_extend,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    output,
                    max_len_extend_tensor,
                )

        # dynamic_symbols = [query_seq_len, kv_seq_len, s]
        mlir_inputs = (
            torch.empty(
                (query_seq_len, num_query_heads, head_size), dtype=torch.float16
            ),
            torch.empty((kv_seq_len, num_kv_heads, head_size), dtype=torch.float16),
            torch.empty((kv_seq_len, num_kv_heads, head_size_kv), dtype=torch.float16),
            torch.empty((kv_seq_len, num_kv_heads, head_size), dtype=torch.float16),
            torch.empty((kv_seq_len, num_kv_heads, head_size_kv), dtype=torch.float16),
            torch.empty((s,), dtype=torch.int32),
            torch.empty((s,), dtype=torch.int32),
            torch.empty((kv_seq_len,), dtype=torch.int32),
            torch.empty(
                (query_seq_len, num_query_heads, head_size_kv), dtype=torch.float16
            ),
            torch.tensor(max_len_extend, dtype=torch.int32),
        )
        e = aot.export(
            WaveExtendAttentionModule(),
            args=mlir_inputs,
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        assert "func.func @main" in mlir_asm
        assert (
            f"stream.executable private @extend_attention__N_Q_N_Q_dyn_H_16_D_Q_128_N_KV_N_KV_dyn_H_KV_1_D_KV_128_S_S_dyn_qkv_input_dtype_f16_indices_input_dtype_i32_output_dtype_f16"
            in mlir_asm
        )
        assert (
            f"func.func private @wave_extend_attention__N_Q_N_Q_dyn_H_16_D_Q_128_N_KV_N_KV_dyn_H_KV_1_D_KV_128_S_S_dyn_qkv_input_dtype_f16_indices_input_dtype_i32_output_dtype_f16"
            in mlir_asm
        )
        assert (
            f"util.func private @wave_extend_attention_N_Q_H_16_D_Q_128_f16_N_KV_H_KV_1_D_Q_128_f16_N_KV_H_KV_1_D_KV_128_f16_N_KV_H_KV_1_D_Q_128_f16_N_KV_H_KV_1_D_KV_128_f16_S_i32_S_i32_N_KV_i32_N_Q_H_16_D_KV_128_f16__i32_N_Q_H_16_D_KV_128_f16"
            in mlir_asm
        )
        mlir_path = tmp_path / "wave_extend_attention.mlir"
        with open(str(mlir_path), "w") as f:
            f.write(mlir_asm)
        vmfb = ireec.compile_file(
            str(mlir_path),
            extra_args=self.hip_flags(),
        )

        instance = ireert.VmInstance()
        devices = [ireert.get_device(iree_flags.iree_device)]
        config = ireert.Config(device=devices[0])
        hal = ireert.create_hal_module(instance, devices=devices)
        binary = ireert.VmModule.copy_buffer(instance, vmfb)
        modules = ireert.load_vm_modules(hal, binary, config=config)

        # Use create_inputs from Wave
