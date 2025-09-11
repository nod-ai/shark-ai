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


@is_mi300x
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
            (512, 512, 3, 16, 128, 1, 128, 458),
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
                q,
                k,
                v,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                max_len_extend,
            ):
                return wave_extend_attention(
                    q,
                    k,
                    v,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    output,
                    max_len_extend,
                )

        # dynamic_symbols = [query_seq_len, kv_seq_len, s]
        e = aot.export(
            WaveExtendAttentionModule(),
            args=(
                torch.empty(
                    (query_seq_len, num_query_heads, head_size), dtype=torch.float16
                ),
                torch.empty((kv_seq_len, num_kv_heads, head_size), dtype=torch.float16),
                torch.empty(
                    (kv_seq_len, num_kv_heads, head_size_kv), dtype=torch.float16
                ),
                torch.empty((kv_seq_len, num_kv_heads, head_size), dtype=torch.float16),
                torch.empty(
                    (kv_seq_len, num_kv_heads, head_size_kv), dtype=torch.float16
                ),
                torch.empty((s), dtype=torch.int32),
                torch.empty((s), dtype=torch.int32),
                torch.empty((kv_seq_len), dtype=torch.int32),
                torch.empty(
                    (query_seq_len, num_query_heads, head_size_kv), dtype=torch.float16
                ),
                torch.tensor(max_len_extend, dtype=torch.int32),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        # assert "func.func @main" in mlir_asm
        # assert (
        #     f"stream.executable private @batched_gemm__B_B_dyn_M_M_dyn_HALF_K_{k//2}_K_OVER_THIRTYTWO_{k//32}_N_{n}_input_dtype_i8_output_dtype_f16"
        #     in mlir_asm
        # )
        # assert (
        #     f"func.func private @wave_mxfp4_bmm__B_B_dyn_M_M_dyn_HALF_K_{k//2}_K_OVER_THIRTYTWO_{k//32}_N_{n}_input_dtype_i8_output_dtype_f16"
        #     in mlir_asm
        # )
        # assert (
        #     f"util.func private @wave_mxfp4_bmm_B_M_HALF_K_{k//2}_i8_B_M_K_OVER_THIRTYTWO_{k//32}_i8_N_{n}_HALF_K_{k//2}_i8_N_{n}_K_OVER_THIRTYTWO_{k//32}_i8_B_M_N_{n}_f16_B_M_N_{n}_f16"
        #     in mlir_asm
        # )
        # mlir_path = tmp_path / "wave_fp4_gemm.mlir"
        # with open(str(mlir_path), "w") as f:
        #     f.write(mlir_asm)
