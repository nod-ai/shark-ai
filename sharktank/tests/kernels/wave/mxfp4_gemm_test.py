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
from sharktank.types.layouts import BlockScaledFp4Layout
from sharktank.types.tensors import unbox_tensor
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_tensor,
)
from sharktank.utils.testing import assert_cosine_similarity_close


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = device_tensor(mxfp4_list, dtype=torch.float32)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


class wave_fp4_gemm(unittest.TestCase):
    def test_fp4_matmul_accuracy(self):
        torch.manual_seed(5)

        # Create float32 inputs
        b = 32
        m = 128
        k = 16384
        n = 16384
        lhs = torch.randn(b, m, k)  # shape: [B, M, K]
        rhs = torch.randn(k, n)  # shape: [K, N]
        expected = lhs @ rhs

        lhs = unbox_tensor(lhs)
        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
        )
        lhs_quantized = quantizer.quantize(lhs)
        lhs_unpacked = lhs_quantized.unpack()
        rhs = unbox_tensor(rhs)
        rhs_quantized = quantizer.quantize(rhs.T)
        rhs_unpacked = rhs_quantized.unpack()
        x = lhs_unpacked.qs_bit_packed
        w_rhs = rhs_unpacked.qs_bit_packed
        flat_x = x.reshape(x.size(0) * x.size(1), x.size(2) * x.size(3))
        flat_w = w_rhs.flatten(start_dim=-2).T
        flat_x_scales = lhs_unpacked.d.reshape(
            lhs_unpacked.d.size(0) * lhs_unpacked.d.size(1), lhs_unpacked.d.size(2)
        )
        w_scales = rhs_unpacked.d

        flat_x_f32 = mxfp4_to_f32(flat_x)
        flat_w_f32 = mxfp4_to_f32(flat_w.T)
        flat_w_f32 = flat_w_f32.T
        flat_x_scales = flat_x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(
            torch.float32
        )
        flat_x_scales_f32 = e8m0_to_f32(flat_x_scales)
        flat_x_f32 = flat_x_f32 * flat_x_scales_f32
        w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
        w_scales_f32 = e8m0_to_f32(w_scales)
        flat_w_f32 = flat_w_f32 * w_scales_f32.T
        torch_flat_out = torch.mm(flat_x_f32, flat_w_f32)
        torch_out = torch_flat_out.view(b, m, n)
        assert_cosine_similarity_close(torch_out, expected, atol=0.1)

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
