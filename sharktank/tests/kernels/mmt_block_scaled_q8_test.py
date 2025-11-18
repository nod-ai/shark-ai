# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from iree.turbine import aot
from amdsharktank import kernels


class mmt_block_scaled_q8_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-3, 1e-5),
            (torch.float32, torch.float16, torch.float32, 1e-3, 1e-5),
            (torch.float16, torch.float16, torch.float32, 1e-3, 1e-5),
        ]
    )
    def testBS32(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        a = torch.rand([4, 16, 3200], dtype=a_dtype) * 64
        d = torch.rand([3200, 100, 1], dtype=d_dtype) * 64
        qs = (torch.rand([3200, 100, 32], dtype=ref_dtype) * 32.0).to(torch.int8)
        result = kernels.mmt_block_scaled_q8(a, d, qs)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        b = (d.to(ref_dtype) * qs.to(ref_dtype)).flatten(1)
        ref = torch.matmul(a.to(ref_dtype), b.T.to(ref_dtype)).to(a_dtype)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, qs):
                return kernels.mmt_block_scaled_q8(a, b, qs)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 32], dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "b": {},
                "qs": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_mmt_block_scaled_q8_3d_3200_3200_32_f32", asm)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, qs):
                return kernels.mmt_block_scaled_q8(a, b, qs)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 32], dtype=torch.float32) * 32.0).to(
                    torch.int8
                ),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_mmt_block_scaled_q8_3d_3200_3200_32_f32", asm)


if __name__ == "__main__":
    unittest.main()
