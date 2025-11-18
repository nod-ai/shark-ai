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


class mmtfp_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32),
            (torch.float16, torch.float16, torch.float16),
            (torch.float16, torch.float32, torch.float16),
            (torch.float32, torch.float16, torch.float32),
        ]
    )
    def test2D(self, a_dtype, b_dtype, ref_dtype):
        a = torch.rand([128, 32], dtype=a_dtype)
        b = torch.rand([256, 32], dtype=b_dtype)
        result = kernels.mmtfp(a, b)
        ref = torch.matmul(a.to(ref_dtype), b.T.to(ref_dtype)).to(a_dtype)
        torch.testing.assert_close(result, ref)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32),
            (torch.float16, torch.float16, torch.float16),
            (torch.float16, torch.float32, torch.float16),
            (torch.float32, torch.float16, torch.float32),
        ]
    )
    def test3DF(self, a_dtype, b_dtype, ref_dtype):
        a = torch.rand([4, 128, 32], dtype=a_dtype)
        b = torch.rand([256, 32], dtype=b_dtype)
        result = kernels.mmtfp(a, b)
        ref = torch.matmul(a.to(ref_dtype), b.T.to(ref_dtype)).to(a_dtype)
        torch.testing.assert_close(result, ref)

    def testExportDynamicDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.mmtfp(a, b)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 128, 32], dtype=torch.float32),
                torch.rand([256, 32], dtype=torch.float32),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "b": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_mmtfp_3d_256_32_f32f32", asm)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.mmtfp(a, b)

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 128, 32], dtype=torch.float32),
                torch.rand([256, 32], dtype=torch.float32),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("@amdsharktank_mmtfp_3d_256_32_f32f32", asm)

    def testExportTooDynamic(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b):
                return kernels.mmtfp(a, b)

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        n = torch.export.Dim("n")
        k = torch.export.Dim("k")
        ep = torch.export.export(
            mod,
            args=(
                torch.rand([4, 128, 32], dtype=torch.float32),
                torch.rand([256, 32], dtype=torch.float32),
            ),
            dynamic_shapes={
                "a": {0: batch, 1: m, 2: k},
                "b": {0: n, 1: k},
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "arg 0 requires a static dim at index 2",
        ):
            aot.export(ep)


if __name__ == "__main__":
    unittest.main()
