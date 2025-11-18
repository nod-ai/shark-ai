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
from iree.turbine.support.conversions import TORCH_DTYPE_TO_IREE_TYPE_ASM
from amdsharktank import kernels
from amdsharktank.utils.testing import skip


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

    def testArgF8AccumF32(self):
        # TODO: make this test not use eager but actually execute with IREE.
        # Does not compile for llvm-cpu with
        # <unknown>:0: error: 'llvm.fpext' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'vector<4xi8>'
        # <unknown>:0: note: see current operation: %120 = "llvm.fpext"(%109) : (vector<4xi8>) -> vector<4xf32>
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

    def testArgF8E4M3FnAccumF32(self):
        # TODO: make this test not use eager but actually execute with IREE.
        # Does not compile for llvm-cpu with
        # <unknown>:0: error: 'llvm.fpext' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'vector<4xi8>'
        # <unknown>:0: note: see current operation: %120 = "llvm.fpext"(%109) : (vector<4xi8>) -> vector<4xf32>
        arg_dtype = torch.float8_e4m3fn
        a = torch.rand([3, 4, 6]).to(arg_dtype)
        b = torch.rand([3, 5, 6]).to(arg_dtype)
        accum_dtype = torch.float32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=accum_dtype), bT.to(dtype=accum_dtype))
        torch.testing.assert_close(result, ref, atol=1e-3, rtol=0)

    def testArgUi8AccumI32(self):
        # TODO: make this test not use eager but actually execute with IREE.
        # Does not work with unsigned types. The kernel needs to be adapted.
        arg_dtype = torch.uint8
        a = ((torch.rand([2, 3, 5]) * 255) + 0.5).to(dtype=arg_dtype)
        b = ((torch.rand([2, 4, 5]) * 255) + 0.5).to(dtype=arg_dtype)
        accum_dtype = torch.int32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=accum_dtype), bT.to(dtype=accum_dtype))
        torch.testing.assert_close(result, ref, atol=0, rtol=0)

    def testArgLhsI8RhsUi8AccumI32(self):
        a = ((torch.rand([2, 3, 5]) - 0.5) * 255).to(dtype=torch.int8)
        b = ((torch.rand([2, 4, 5]) * 255) + 0.5).to(dtype=torch.uint8)
        accum_dtype = torch.int32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=accum_dtype), bT.to(dtype=accum_dtype))
        torch.testing.assert_close(result, ref, atol=0, rtol=0)

    def testArgI8AccumI32(self):
        arg_dtype = torch.int8
        a = ((torch.rand([2, 3, 5]) - 0.5) * 255).to(dtype=arg_dtype)
        b = ((torch.rand([2, 3, 5]) - 0.5) * 255).to(dtype=arg_dtype)
        accum_dtype = torch.int32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=accum_dtype), bT.to(dtype=accum_dtype))
        torch.testing.assert_close(result, ref, atol=0, rtol=0)

    @pytest.mark.xfail(
        reason="""No uint32 dtype conversions in IREE Turbine.
        Does not work with unsigned types. The kernel needs to be adapted.
        The problem is that we reinterpret cast to signless integer types.
        Maybe linalg.batch_matmul when promoting from i8 to i32 assumes a
        signed type even though i8 is signless."""
    )
    def testArgUi8AccumUi32(self):
        arg_dtype = torch.uint8
        a = ((torch.rand([2, 3, 5]) * 255) + 0.5).to(dtype=arg_dtype)
        b = ((torch.rand([2, 4, 5]) * 255) + 0.5).to(dtype=arg_dtype)
        accum_dtype = torch.uint32
        result = kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        bT = torch.transpose(b, 1, 2)
        ref = torch.matmul(a.to(dtype=torch.int32), bT.to(dtype=torch.int32))
        ref = ref.to(dtype=accum_dtype)
        torch.testing.assert_close(result, ref, atol=0, rtol=0)

    @parameterized.expand(
        [
            (torch.int32, None),
            (torch.float8_e4m3fnuz, torch.float32),
        ]
    )
    def testExportStaticDims(
        self, arg_dtype: torch.dtype, accum_dtype: torch.dtype | None
    ):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.batch_matmul_transpose_b(a, b, accum_dtype=accum_dtype)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                (torch.rand([4, 16, 2]) * 64).to(arg_dtype),
                (torch.rand([4, 8, 2]) * 64).to(arg_dtype),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        arg_dtype_asm = TORCH_DTYPE_TO_IREE_TYPE_ASM[arg_dtype]
        accum_dtype_asm = arg_dtype_asm
        if accum_dtype is not None:
            accum_dtype_asm = TORCH_DTYPE_TO_IREE_TYPE_ASM[accum_dtype]
        self.assertIn(
            (
                "@amdsharktank_batch_matmul_transpose_b_"
                f"L4x16x2x{arg_dtype_asm}_R4x8x2x{arg_dtype_asm}_{accum_dtype_asm}"
            ),
            asm,
        )


if __name__ == "__main__":
    unittest.main()
