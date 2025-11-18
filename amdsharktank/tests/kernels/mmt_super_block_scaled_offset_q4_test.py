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


class mmt_super_block_scaled_offset_q4_unsigned(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, torch.float32, torch.float32, 1e-2, 1e-3),
            (torch.float32, torch.float16, torch.float32, 1e-2, 1e-3),
            (torch.float16, torch.float16, torch.float32, 1e-2, 1e-3),
        ]
    )
    @unittest.skip(
        "compiler bad tile selection. fixed by: "
        "https://github.com/iree-org/iree/pull/17115 "
        "TODO: There is still a numeric bug in the implementation. Triage once lands."
    )
    def test_basic(self, a_dtype, d_dtype, ref_dtype, atol, rtol):
        # n = 2560, k = 5120, sup = 20, sub = 8, bs = 32
        a = torch.rand([4, 16, 5120], dtype=a_dtype) / 256.0
        d = torch.rand([2560, 20, 1], dtype=d_dtype) / 256.0
        dmin = torch.rand([2560, 20, 1], dtype=d_dtype) * 5.0
        sb_scales_hi = (torch.rand([2560, 20, 2], dtype=d_dtype) * 127).to(torch.uint8)
        sb_scales_low = (torch.rand([2560, 20, 4], dtype=d_dtype) * 127).to(torch.uint8)
        sb_mins_hi = (torch.rand([2560, 20, 2], dtype=d_dtype) * 127).to(torch.uint8)
        sb_mins_low = (torch.rand([2560, 20, 4], dtype=d_dtype) * 127).to(torch.uint8)
        qs = (torch.rand([2560, 20, 8, 16], dtype=torch.float32) * 255.0).to(
            torch.uint8
        )
        result = kernels.mmt_super_block_scaled_offset_q4_unsigned(
            a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs
        )

        ref_qs = layout_utils.promote_linear_i4_block_to_i8(qs)
        ref_sb_scales = layout_utils.promote_linear_i6_block_to_i8(
            sb_scales_hi, sb_scales_low
        )
        ref_sb_mins = layout_utils.promote_linear_i6_block_to_i8(
            sb_mins_hi, sb_mins_low
        )
        ref_d_scaled = (d.to(ref_dtype) * ref_sb_scales.to(ref_dtype)).unsqueeze(-1)
        ref_dmin_scaled = (dmin.to(ref_dtype) * ref_sb_mins.to(ref_dtype)).unsqueeze(-1)
        ref_b = (ref_d_scaled * ref_qs.to(ref_dtype) - ref_dmin_scaled).flatten(1)
        ref = torch.matmul(a.to(ref_dtype), ref_b.T.to(ref_dtype))
        torch.testing.assert_close(result.to(ref_dtype), ref, atol=atol, rtol=rtol)

    def testExportDynamicDims(self):
        # n = 2560, k = 5120, sup = 20, sub = 8, bs = 32
        a = torch.rand([4, 16, 5120], dtype=torch.float32)
        d = torch.rand([2560, 20, 1], dtype=torch.float16)
        dmin = torch.rand([2560, 20, 1], dtype=torch.float16)
        sb_scales_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_scales_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        qs = (torch.rand([2560, 20, 8, 16], dtype=torch.float32) * 127).to(torch.uint8)

        class MyModule(torch.nn.Module):
            def forward(
                self,
                a,
                d,
                dmin,
                sb_scales_hi,
                sb_scales_low,
                sb_mins_hi,
                sb_mins_low,
                qs,
            ):
                return kernels.mmt_super_block_scaled_offset_q4_unsigned(
                    a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs
                )

        mod = MyModule()
        batch = torch.export.Dim("batch")
        m = torch.export.Dim("m")
        ep = torch.export.export(
            mod,
            args=(a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs),
            dynamic_shapes={
                "a": {0: batch, 1: m},
                "d": {},
                "dmin": {},
                "sb_scales_hi": {},
                "sb_scales_low": {},
                "sb_mins_hi": {},
                "sb_mins_low": {},
                "qs": {},
            },
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@mmt_super_block_scaled_offset_q4_unsigned_3d_2560_5120_20_8_32_f32", asm
        )

    def testExportStaticDims(self):
        # n = 2560, k = 5120, sup = 20, sub = 8, bs = 32
        a = torch.rand([4, 16, 5120], dtype=torch.float32)
        d = torch.rand([2560, 20, 1], dtype=torch.float16)
        dmin = torch.rand([2560, 20, 1], dtype=torch.float16)
        sb_scales_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_scales_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_hi = (torch.rand([2560, 20, 2], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        sb_mins_low = (torch.rand([2560, 20, 4], dtype=torch.float32) * 127).to(
            torch.uint8
        )
        qs = (torch.rand([2560, 20, 8, 16], dtype=torch.float32) * 127).to(torch.uint8)

        class MyModule(torch.nn.Module):
            def forward(
                self,
                a,
                d,
                dmin,
                sb_scales_hi,
                sb_scales_low,
                sb_mins_hi,
                sb_mins_low,
                qs,
            ):
                return kernels.mmt_super_block_scaled_offset_q4_unsigned(
                    a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs
                )

        mod = MyModule()
        ep = torch.export.export(
            mod,
            args=(a, d, dmin, sb_scales_hi, sb_scales_low, sb_mins_hi, sb_mins_low, qs),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@mmt_super_block_scaled_offset_q4_unsigned_3d_2560_5120_20_8_32_f32", asm
        )


if __name__ == "__main__":
    unittest.main()
