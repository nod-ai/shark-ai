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
from amdsharktank.types import layout_utils


class mmt_block_scaled_offset_q4_unsigned_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    def test_basic(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        a = torch.rand([4, 16, 3200], dtype=a_dtype) / 256.0
        d = torch.rand([3200, 100, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([3200, 100, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=d_dtype) + 16.0
        result = kernels.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(1)
        ref = torch.matmul(a.to(ref_dtype), b.T.to(ref_dtype))
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return kernels.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 16], dtype=torch.float32) * 32).to(torch.uint8),
                torch.rand([3200, 100, 1], dtype=torch.float16),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "d": {},
                "qs": {},
                "m": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@amdsharktank_mmt_block_scaled_offset_q4_unsigned_3d_3200_3200_32_f32", asm
        )

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return kernels.mmt_block_scaled_offset_q4_unsigned(a, d, qs, m)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 3200], dtype=torch.float32),
                torch.rand([3200, 100, 1], dtype=torch.float16),
                (torch.rand([3200, 100, 16], dtype=torch.float32) * 32).to(torch.uint8),
                torch.rand([3200, 100, 1], dtype=torch.float16),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@amdsharktank_mmt_block_scaled_offset_q4_unsigned_3d_3200_3200_32_f32", asm
        )


if __name__ == "__main__":
    unittest.main()
