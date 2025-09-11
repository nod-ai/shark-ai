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
from sharktank.kernels.wave.utils import create_extend_attention_inputs, ref_extend_attn
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from dataclasses import replace
from torch.testing import assert_close


# @is_mi300x
@pytest.mark.usefixtures("iree_flags")
class TestExtendAttention:
    def hip_flags(self):
        return [
            "--iree-hip-target=gfx950",
            "--iree-hal-target-device=hip",
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
        "query_seq_len, kv_seq_len, num_query_heads, head_size, num_kv_heads, head_size_kv, max_len_extend, is_causal, use_custom_mask",
        [
            (879, 879, 16, 128, 1, 128, 458, False, False),
        ],
    )
    def test_extend_attention_export_compile_run(
        self,
        iree_flags: IreeFlags,
        tmp_path: Path,
        query_seq_len: int,
        kv_seq_len: int,
        num_query_heads: int,
        head_size: int,
        num_kv_heads: int,
        head_size_kv: int,
        max_len_extend: int,
        is_causal: bool,
        use_custom_mask: bool,
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

        # Use create_inputs from Wave
        shape = AttentionShape(
            context_len=1024,
            num_seqs=2,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            query_seq_len=query_seq_len,
            head_size_kv=head_size_kv,
            head_size=head_size,
            kv_seq_len=kv_seq_len,
            max_seq_len=max_len_extend,
        )
        dtype = torch.float16
        torch.manual_seed(0)
        (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_seq_len,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            b_start_loc,
            b_seq_len_prefix,
            extend_token_num,
            max_len_extend_wave,
            logit_cap,
            _,
            _,
        ) = create_extend_attention_inputs(shape, dtype)
        shape = replace(shape, max_seq_len=max_len_extend_wave)
        output = torch.empty(
            extend_token_num, shape.num_query_heads, shape.head_size, dtype=dtype
        )

        # dynamic_symbols = [query_seq_len, kv_seq_len, s]
        mlir_inputs = (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            output,
            torch.tensor(max_len_extend_wave, dtype=torch.int32),
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

        _wave_extend_attention_main = modules[-1].main
        iree_results = _wave_extend_attention_main(*mlir_inputs)
        iree_results = torch.from_numpy(
            np.asarray(iree_results.to_host()).astype(np.float32)
        )
        ref_output = ref_extend_attn(
            q_extend=q_extend,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            b_req_idx=b_req_idx,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            b_seq_len_prefix=b_seq_len_prefix,
            max_len_extend=max_len_extend,
            extend_token_num=extend_token_num,
            dtype=dtype,
            is_causal=(
                is_causal or use_custom_mask
            ),  # Custom mask is set to a causal mask.
            logit_cap=logit_cap,
        )

        assert_close(iree_results, ref_output, rtol=1e-3, atol=1e-3, check_dtype=False)
