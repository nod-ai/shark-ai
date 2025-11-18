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
from amdsharktank.kernels.wave.mxfp4_gemm import wave_mxfp4_bmm
from parameterized import parameterized
from amdsharktank.types.quantizers import DynamicFp4BlockQuantizer
from amdsharktank.types.tensors import unbox_tensor
from amdsharktank.utils.testing import assert_cosine_similarity_close
import iree.compiler as ireec
import iree.runtime as ireert
from pathlib import Path
import numpy as np
from amdsharktank.utils.testing import is_mi350x, IreeFlags


@is_mi350x
@pytest.mark.usefixtures("iree_flags")
class TestWaveFp4Gemm:
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
            "--iree-hip-encoding-layout-resolver=data-tiling",
        ]

    @is_mi350x
    @pytest.mark.parametrize(
        "b, m, k, n",
        [
            (2, 256, 512, 128),
            (1, 512, 1024, 256),
        ],
    )
    def test_wave_fp4_gemm_export_compile_run(
        self,
        deterministic_random_seed,
        iree_flags: IreeFlags,
        tmp_path: Path,
        b: int,
        m: int,
        k: int,
        n: int,
    ):
        assert k % 32 == 0

        class WaveMxfp4Module(torch.nn.Module):
            def forward(self, x, x_scales, w_t, w_scales, output):
                return wave_mxfp4_bmm(x, x_scales, w_t, w_scales, output)

        e = aot.export(
            WaveMxfp4Module(),
            args=(
                torch.empty((b, m, k // 2), dtype=torch.uint8),
                torch.empty((b, m, k // 32), dtype=torch.uint8),
                torch.empty((n, k // 2), dtype=torch.uint8),
                torch.empty((n, k // 32), dtype=torch.uint8),
                torch.empty((b, m, n), dtype=torch.float16),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        assert "func.func @main" in mlir_asm
        assert (
            f"stream.executable private @batched_gemm__B_B_dyn_M_M_dyn_HALF_K_{k//2}_K_OVER_THIRTYTWO_{k//32}_N_{n}_input_dtype_i8_output_dtype_f16"
            in mlir_asm
        )
        assert (
            f"func.func private @wave_mxfp4_bmm__B_B_dyn_M_M_dyn_HALF_K_{k//2}_K_OVER_THIRTYTWO_{k//32}_N_{n}_input_dtype_i8_output_dtype_f16"
            in mlir_asm
        )
        assert (
            f"util.func private @wave_mxfp4_bmm_B_M_HALF_K_{k//2}_i8_B_M_K_OVER_THIRTYTWO_{k//32}_i8_N_{n}_HALF_K_{k//2}_i8_N_{n}_K_OVER_THIRTYTWO_{k//32}_i8_B_M_N_{n}_f16_B_M_N_{n}_f16"
            in mlir_asm
        )
        mlir_path = tmp_path / "wave_fp4_gemm.mlir"
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

        lhs = torch.randn(b, m, k, dtype=torch.float32)  # shape: [B, M, K]
        rhs = torch.randn(k, n, dtype=torch.float32)  # shape: [K, N]
        expected = lhs @ rhs

        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
        )
        lhs_quantized = quantizer.quantize(lhs)
        lhs_unpacked = lhs_quantized.unpack()
        rhs_quantized = quantizer.quantize(rhs.mT)
        rhs_unpacked = rhs_quantized.unpack()

        x = lhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        x_scales = lhs_unpacked.d.squeeze(-1)
        w_t = rhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        w_scales = rhs_unpacked.d.squeeze(-1)
        output = torch.empty(
            [lhs.shape[0], lhs.shape[1], rhs_unpacked.shape[0]],
            dtype=torch.float16,
        )
        _wave_fp4_gemm_main = modules[-1].main
        iree_results = _wave_fp4_gemm_main(x, x_scales, w_t, w_scales, output)
        iree_results = torch.from_numpy(
            np.asarray(iree_results.to_host()).astype(np.float32)
        )
        assert_cosine_similarity_close(iree_results, expected, atol=0.05)
