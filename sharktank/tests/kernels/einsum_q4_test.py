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


class einsum_2args_q4_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    def test_basic_mk_menk_men(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        a = torch.rand([2, 320], dtype=a_dtype) / 256.0
        d = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([2, 4, 8, 10, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) + 16.0
        einsum_string = "mk,menk->men"
        result = kernels.einsum_2args_q4(a, d, qs, m, einsum_string)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(3)
        ref = torch.einsum(einsum_string, a.to(ref_dtype), b.to(ref_dtype))
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    def test_basic_mek_menk_men(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        a = torch.rand([2, 4, 320], dtype=a_dtype) / 256.0
        d = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([2, 4, 8, 10, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([2, 4, 8, 10, 1], dtype=d_dtype) + 16.0
        einsum_string = "mek,menk->men"
        result = kernels.einsum_2args_q4(a, d, qs, m, einsum_string)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(3)
        ref = torch.einsum(einsum_string, a.to(ref_dtype), b.to(ref_dtype))
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    def test_basic_me_men_men(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        a = torch.rand([2, 4], dtype=a_dtype) / 256.0
        d = torch.rand([2, 4, 10, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([2, 4, 10, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([2, 4, 10, 1], dtype=d_dtype) + 16.0
        einsum_string = "me,men->men"
        result = kernels.einsum_2args_q4(a, d, qs, m, einsum_string)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        qs_i8 = layout_utils.promote_linear_i4_block_to_i8(qs)
        b = (d.to(ref_dtype) * qs_i8.to(ref_dtype) + m.to(ref_dtype)).flatten(2)
        ref = torch.einsum(einsum_string, a.to(ref_dtype), b.to(ref_dtype))
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return kernels.einsum_2args_q4(a, d, qs, m, "ij,jk->ik")

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([16, 320], dtype=torch.float32),
                torch.rand([320, 2, 1], dtype=torch.float16),
                (torch.rand([320, 2, 16], dtype=torch.float32) * 32).to(torch.uint8),
                torch.rand([320, 2, 1], dtype=torch.float16),
            ),
            dynamic_shapes={
                "a": {},
                "d": {},
                "qs": {},
                "m": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_einsum_2args_q4_ij_jk_ik_32_f32", asm)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                return kernels.einsum_2args_q4(a, d, qs, m, "mek,menk->men")

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 16, 320], dtype=torch.float32),
                torch.rand([4, 16, 2, 10, 1], dtype=torch.float16),
                (torch.rand([4, 16, 2, 10, 16], dtype=torch.float32) * 32).to(
                    torch.uint8
                ),
                torch.rand([4, 16, 2, 10, 1], dtype=torch.float16),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_einsum_2args_q4_mek_menk_men_32_f32", asm)


if __name__ == "__main__":
    unittest.main()
