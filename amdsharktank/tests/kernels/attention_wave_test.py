# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
import pytest
import sys

import torch
from iree.compiler.passmanager import PassManager
from iree.compiler.ir import Context, Module
import iree.turbine.aot as aot
from amdsharktank.kernels.wave.attention import wave_bhsd_flash_attention
from parameterized import parameterized


class wave_attention(unittest.TestCase):
    @pytest.mark.xfail(
        sys.platform == "win32",
        reason="Unable to pasrse attribute on windows due to escaped path issue. Issue: https://github.com/nod-ai/amdshark-ai/issues/2648",
    )
    def test_wave_attention_causal(self):
        class WaveBhsdModule(torch.nn.Module):
            def forward(self, q, k, v, output):
                return wave_bhsd_flash_attention(q, k, v, output)

        e = aot.export(
            WaveBhsdModule(),
            args=(
                torch.empty((4, 32, 128, 128), dtype=torch.float16),
                torch.empty((4, 32, 128, 128), dtype=torch.float16),
                torch.empty((4, 32, 128, 128), dtype=torch.float16),
                torch.empty((4, 32, 128, 128), dtype=torch.float32),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        self.assertIn(
            ("func.func @main"),
            mlir_asm,
        )
        self.assertIn(
            ("stream.executable private @base_attention"),
            mlir_asm,
        )
        self.assertIn(
            ("func.func private @wave_flash_attention_4_32_128_128_f16_f32"),
            mlir_asm,
        )
        self.assertIn(
            (
                "util.func private @wave_bhsd_flash_attention_B_4_H_32_M_128_K1_128_f16_B_4_H_32_K2_128_K1_128_f16_B_4_H_32_K2_128_N_128_f16_B_4_H_32_M_128_N_128_f32_B_4_H_32_M_128_N_128_f32"
            ),
            mlir_asm,
        )


if __name__ == "__main__":
    unittest.main()
