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
from sharktank import kernels
from sharktank.types import layout_utils
from sharktank.utils import debugging
from sharktank import ops
from sharktank.ops.signatures import scaled_dot_product_attention


class wave_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1e-3, 1e-5),
        ]
    )
    def testWaveAttentionCausal(self, atol, rtol):
        dtype = torch.float16
        accum_dtype = torch.float32
        q = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        k = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        v = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        output = (torch.zeros([4, 32, 128, 128])).to(accum_dtype)
        result = kernels.wave_flash_attention(q, k, v, output)
        breakpoint()

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
