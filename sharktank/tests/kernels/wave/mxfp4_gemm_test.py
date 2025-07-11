# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch
from iree.compiler.passmanager import PassManager
from iree.compiler.ir import Context, Module
import iree.turbine.aot as aot
from sharktank.kernels.wave.mxfp4_gemm import wave_mxfp4_bmm
from parameterized import parameterized
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
from sharktank.types.tensors import unbox_tensor


class wave_fp4_gemm(unittest.TestCase):
    def test_wave_fp4_gemm_bmm(self):
        # BMK @ NK = BMN
        lhs = torch.randint(low=0, high=256, size=(4, 1024, 1024), dtype=torch.float32)
        rhs = torch.randint(low=0, high=256, size=(1024, 1024), dtype=torch.float32)
        lhs = unbox_tensor(lhs)
        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
        )
        lhs_quantized = quantizer.quantize(lhs)
        lhs_unpacked = lhs_quantized.unpack()
        rhs = unbox_tensor(rhs)
        rhs_quantized = quantizer.quantize(rhs)
        rhs_unpacked = rhs_quantized.unpack()
        output = torch.zeros(
            [lhs.shape[0], lhs.shape[1], rhs.shape[0]],
            dtype=torch.float32,
        )
        # wave_output = wave_mxfp4_bmm(
        #     lhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        #     lhs_unpacked.d,
        #     rhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        #     rhs_unpacked.d,
        #     output,
        # )

    def test_wave_fp4_gemm(self):
        class WaveMxfp4Module(torch.nn.Module):
            def forward(self, x, x_scales, w_t, w_scales, output):
                return wave_mxfp4_bmm(x, x_scales, w_t, w_scales, output)

        e = aot.export(
            WaveMxfp4Module(),
            args=(
                torch.empty((4, 1024, 512), dtype=torch.uint8),
                torch.empty((4, 1024, 32), dtype=torch.uint8),
                torch.empty((1024, 512), dtype=torch.uint8),
                torch.empty((1024, 32), dtype=torch.uint8),
                torch.empty((4, 1024, 1024), dtype=torch.float32),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        with open("test.mlir", "w") as f:
            f.write(mlir_asm)
        self.assertIn(
            ("func.func @main"),
            mlir_asm,
        )
        self.assertIn(
            ("stream.executable private @batched_gemm"),
            mlir_asm,
        )
        self.assertIn(
            (
                "func.func private @wave_mxfp4_bmm_B_dyn_M_dyn_HALF_K_512_u8_B_dyn_M_dyn_K_OVER_THIRTYTWO_32_u8_N_1024_HALF_K512_u8_N_1024_K_OVER_THIRTYTWO_32_u8_B_dyn_M_dyn_N_1024_f32"
            ),
            mlir_asm,
        )
        self.assertIn(
            (
                "util.func private @wave_mxfp4_bmm_B_M_HALF_K_512_i8_B_M_K_OVER_THIRTYTWO_32_i8_N_1024_HALF_K_512_i8_N_1024_K_OVER_THIRTYTWO_32_i8_B_M_N_1024_f32_B_M_N_1024_f32"
            ),
            mlir_asm,
        )


if __name__ == "__main__":
    unittest.main()
