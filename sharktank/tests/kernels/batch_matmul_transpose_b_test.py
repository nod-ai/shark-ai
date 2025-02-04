# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized
import pytest
import torch

from iree.turbine import aot
from sharktank import kernels
from sharktank.utils.testing import skip


class batch_matmul_transpose_b_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1e-3, 1e-5),
            (1e-3, 1e-5),
            (1e-3, 1e-5),
        ]
    )
    def testBS32(self, atol, rtol):
        dtype = torch.int32
        a = (torch.rand([4, 16, 3200]) * 64).to(dtype)
        b = (torch.rand([4, 8, 3200]) * 64).to(dtype)
        result = kernels.batch_matmul_transpose_b(a, b)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a, bT)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    @pytest.mark.xfail(
        reason="""Does not compile for llvm-cpu with
          <unknown>:0: error: 'llvm.fpext' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'vector<4xi8>'
          <unknown>:0: note: see current operation: %120 = "llvm.fpext"(%109) : (vector<4xi8>) -> vector<4xf32>
          """
    )
    def testArgF8AccumF32(self):
        arg_dtype = torch.float8_e4m3fnuz
        a = torch.rand([3, 4, 6]).to(arg_dtype)
        b = torch.rand([3, 5, 6]).to(arg_dtype)
        accum_dtype = torch.float32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=accum_dtype), bT.to(dtype=accum_dtype))
        torch.testing.assert_close(result, ref, atol=1e-3, rtol=0)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.batch_matmul_transpose_b(a, b)

        mod = MyModule()
        dtype = torch.int32
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([4, 16, 2]) * 64).to(dtype),
                (torch.rand([4, 8, 2]) * 64).to(dtype),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@sharktank_batch_matmul_transpose_b_L4x16x2xi32_R4x8x2xi32_i32", asm
        )

    def testExportArgF8AccumF32(self):
        accum_dtype = torch.float32
        arg_type = torch.float8_e4m3fnuz

        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([4, 16, 2])).to(arg_type),
                (torch.rand([4, 8, 2])).to(arg_type),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@sharktank_batch_matmul_transpose_b_L4x16x2xf8E4M3FNUZ_R4x8x2xf8E4M3FNUZ_f32",
            asm,
        )


if __name__ == "__main__":
    unittest.main()
