# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import torch
import unittest

from amdsharktank import kernels
from amdsharktank import ops


class rotary_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rotary(self):
        dtype = torch.float32
        a = torch.rand([1, 128, 1, 64], dtype=dtype)
        rot = torch.rand([128, 32], dtype=dtype)
        res_b = ops.view_as_real(torch.complex(rot, rot))
        ref_b = torch.complex(torch.cos(rot), torch.sin(rot))

        result = kernels.apply_rotary_embedding(a, res_b.unsqueeze(0))
        ref = ops.view_as_real(ops.view_as_complex(a) * ref_b[None, :, None, :])
        torch.testing.assert_close(result, ref)
