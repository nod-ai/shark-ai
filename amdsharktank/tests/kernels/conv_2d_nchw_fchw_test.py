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
import torch.nn.functional as F

from iree.turbine import aot
from amdsharktank import kernels


class conv_2d_nchw_fchw_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.int8, "int32", 1e-3, 1e-5),
            (torch.int8, "int8", 1e-3, 1e-5),
            (torch.float16, "float16", 1e-1, 1e-1),  # Different accumulators from ref
            (torch.float16, "float32", 1e-3, 1e-5),
            (torch.float32, "float32", 1e-3, 1e-5),
        ]
    )
    def testBS32(self, input_dtype, output_dtype_name, atol, rtol):
        output_dtype = getattr(torch, output_dtype_name)
        inputs = (torch.rand([2, 4, 64, 64]) * 64).to(input_dtype)
        padding = [1, 1]
        extended_list = [item for item in padding for _ in range(2)]
        inputs_pad = F.pad(inputs, pad=extended_list)

        weights = (torch.rand([8, 4, 3, 3]) * 64).to(input_dtype)
        bias = (torch.rand([8]) * 64).to(dtype=output_dtype)
        result = kernels.conv_2d_nchw_fchw(
            inputs_pad, weights, bias, [1, 1], [1, 1], output_dtype=output_dtype_name
        )

        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.nn.functional.conv2d(
            inputs.to(dtype=output_dtype),
            weights.to(dtype=output_dtype),
            bias=bias.to(dtype=output_dtype),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        print(result.shape)
        print(ref.shape)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    def testExportStaticDims(self):
        class MyModule(torch.nn.Module):
            def forward(self, a, b, c):
                return kernels.conv_2d_nchw_fchw(
                    a, b, c, [1, 1], [1, 1], output_dtype="int32"
                )

        mod = MyModule()
        dtype = torch.int8
        inputs = torch.rand([2, 320, 64, 64]) * 64
        padding = [1, 1]
        extended_list = [item for item in padding for _ in range(2)]
        inputs_pad = F.pad(inputs, pad=extended_list)
        ep = torch.export.export(
            mod,
            args=(
                (inputs_pad).to(dtype),
                (torch.rand([640, 320, 3, 3]) * 64).to(dtype),
                (torch.rand([640]) * 64).to(torch.int32),
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn(
            "@amdsharktank_conv_2d_nchw_fchw_I2x320x66x66xi8_W640x320x3x3xi8_B640xi32_Oi32_S1x1_D1x1",
            asm,
        )


if __name__ == "__main__":
    unittest.main()
