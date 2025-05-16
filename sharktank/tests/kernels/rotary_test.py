# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import torch
import unittest
from parameterized import parameterized

from sharktank import kernels
from sharktank import ops


class rotary_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1, 16, 1, 32),
            (1, 16, 1, 7),
            (1, 16, 64, 1),
            (4, 16, 64, 32),
        ]
    )
    def test_rotary(self, bs, heads, seq_len, dims):
        dtype = torch.float32
        a = torch.rand([bs, heads, seq_len, dims * 2], dtype=dtype)
        rot = torch.rand([bs, heads, dims], dtype=dtype)
        res_b = ops.view_as_real(torch.complex(rot, rot))
        ref_b = torch.complex(torch.cos(rot), torch.sin(rot))

        result = kernels.apply_rotary_embedding(a, res_b)
        ref = ops.view_as_real(ops.view_as_complex(a) * ref_b.unsqueeze(2))
        torch.testing.assert_close(result, ref)

    @parameterized.expand([(1, 1), (4, 1), (1, 64), (4, 64)])
    def test_upcast(self, bs, seq_len):
        narrow_dtype = torch.float16
        wide_dtype = torch.float32
        a = torch.rand([bs, 128, seq_len, 64], dtype=narrow_dtype)
        rot = torch.rand([bs, 128, 32], dtype=wide_dtype)
        res_b = ops.view_as_real(torch.complex(rot, rot))
        ref_b = torch.complex(torch.cos(rot), torch.sin(rot))

        result = kernels.apply_rotary_embedding(a, res_b, dtype=wide_dtype)
        ref = ops.view_as_real(
            ops.view_as_complex(ops.to(a, dtype=wide_dtype)) * ref_b.unsqueeze(2)
        )
        torch.testing.assert_close(result, ref)


if __name__ == "__main__":
    unittest.main()
