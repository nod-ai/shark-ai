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
from sharktank.utils.testing import assert_cosine_similarity_close
import iree.compiler as ireec
import iree.runtime as ireert
from pathlib import Path
import numpy as np


mlir_filename = "wave_fp4_gemm.mlir"
class wave_fp4_gemm(unittest.TestCase):
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

    def test_1_wave_fp4_gemm_export(self):
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
                torch.empty((4, 1024, 1024), dtype=torch.float16),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        self.assertIn(
            ("func.func @main"),
            mlir_asm,
        )
        self.assertIn(
            ("stream.executable private @batched_gemm__B_B_dyn_M_M_dyn_HALF_K_512_K_OVER_THIRTYTWO_32_N_1024_input_dtype_i8_output_dtype_f16"),
            mlir_asm,
        )
        self.assertIn(
            (
                "func.func private @wave_mxfp4_bmm__B_B_dyn_M_M_dyn_HALF_K_512_K_OVER_THIRTYTWO_32_N_1024_input_dtype_i8_output_dtype_f16"
            ),
            mlir_asm,
        )
        self.assertIn(
            (
                "util.func private @wave_mxfp4_bmm_B_M_HALF_K_512_i8_B_M_K_OVER_THIRTYTWO_32_i8_N_1024_HALF_K_512_i8_N_1024_K_OVER_THIRTYTWO_32_i8_B_M_N_1024_f16_B_M_N_1024_f16"
            ),
            mlir_asm,
        )
        with open(mlir_filename, 'w') as f:
            f.write(mlir_asm)

    
    def test_2_wave_fp4_gemm_compile_run_and_compare(self):
        torch.manual_seed(5)
        self.assertTrue(Path(mlir_filename).exists(), "Missing MLIR from export step.")
        vmfb = ireec.compile_file(
            mlir_filename,
            extra_args=self.hip_flags(),
        )
        self.instance = ireert.VmInstance()
        self.devices = [
            ireert.get_device('hip://0')
        ]
        self.config = ireert.Config(device=self.devices[0])
        self.hal = ireert.create_hal_module(self.instance, devices=self.devices)
        self.binary = ireert.VmModule.copy_buffer(self.instance, vmfb)
        self.modules = ireert.load_vm_modules(
            self.hal, self.binary, config=self.config
        )

        # Create float32 inputs
        b = 4
        m = 1024
        k = 1024
        n = 1024
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
        rhs_quantized = quantizer.quantize(rhs)
        rhs_unpacked = rhs_quantized.unpack()

        x = lhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        x_scales = lhs_unpacked.d.squeeze(-1)
        w_t = rhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        w_scales = rhs_unpacked.d.squeeze(-1)
        output = torch.zeros(
            [lhs.shape[0], lhs.shape[1], rhs_unpacked.shape[0]],
            dtype=torch.float16,
        )
        self._wave_fp4_gemm_main = self.modules[-1].main
        iree_results = self._wave_fp4_gemm_main(x, x_scales, w_t, w_scales, output)
        iree_results = torch.from_numpy(np.asarray(iree_results).astype(np.float32))
        assert_cosine_similarity_close(iree_results, expected, atol=0.05)


if __name__ == "__main__":
    unittest.main()
