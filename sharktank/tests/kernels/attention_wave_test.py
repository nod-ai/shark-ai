# Copyright 2025 Advanced Micro Devices, Inc.
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
from iree.turbine.kernel.wave.common.utils import scaled_dot_product_attention_bhsd


class wave_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1e-3, 1e-3),
        ]
    )
    def testWaveAttentionCausal(self, atol, rtol):
        dtype = torch.float16
        accum_dtype = torch.float32
        q = torch.randn([4, 32, 128, 128], device="cuda", dtype=dtype)
        k = torch.randn([4, 32, 128, 128], device="cuda", dtype=dtype)
        v = torch.randn([4, 32, 128, 128], device="cuda", dtype=dtype)
        output = torch.zeros([4, 32, 128, 128], device="cuda", dtype=accum_dtype)
        result = kernels.wave_flash_attention(q, k, v, output)

        # Tolerances are empirical and results are not expected to match exactly.
        ref = scaled_dot_product_attention_bhsd(q, k, v, is_causal=True)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
