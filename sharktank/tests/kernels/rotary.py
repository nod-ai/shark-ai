# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import torch
import unittest

from sharktank import kernels
from sharktank import ops


class rotary_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rotary_interleaved(self):
        dtype = torch.float32
        a = torch.rand([1, 128, 1, 64], dtype=dtype)
        rot = torch.rand([128, 32], dtype=dtype)

        ref_a = ops.view_as_complex(a)
        ref_b = torch.complex(torch.cos(rot), torch.sin(rot))
        ref = ops.view_as_real(ref_a * ref_b[None, :, None, :])

        res_a = a
        res_b = ops.view_as_real(torch.complex(rot, rot)).unsqueeze(0)
        res = kernels.apply_rotary_embedding(res_a, res_b, mode="interleave")

        torch.testing.assert_close(res, ref)

    def test_rotary_concatted(self):
        dtype = torch.float32
        a = torch.rand([1, 128, 1, 64], dtype=dtype)
        rot = torch.rand([128, 32], dtype=dtype)

        ref_a = ops.view_as_complex(a)
        ref_b = torch.complex(torch.cos(rot), torch.sin(rot))
        ref = ops.view_as_real(ref_a * ref_b[None, :, None, :])

        res_a = a.unflatten(-1, (-1, 2)).transpose(-2, -1).flatten(-2, -1)
        res_b = torch.concat((rot, rot), dim=-1).unsqueeze(0)
        res = kernels.apply_rotary_embedding(res_a, res_b, mode="concat")
        res = res.unflatten(-1, (2, -1)).transpose(-2, -1).flatten(-2, -1)

        torch.testing.assert_close(res, ref)
