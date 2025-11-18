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


class topk_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_topk(self):
        dtype = torch.float32
        # Create input tensor with shape [8, 1, 128] (batch_size=8, seq_len=1, dim=128)
        x = torch.rand([8, 10], dtype=dtype)

        # Add some known values to verify topk behavior
        x[0, :4] = torch.tensor([10.0, 5.0, 8.0, 3.0], dtype=dtype)

        # Get reference result using torch.topk
        ref_values, ref_indices = torch.topk(x, k=4, dim=-1)

        # Get result from our kernel
        y = torch.arange(10)[None, :].repeat(8, 1).to(torch.int32)
        result = kernels.iree_topk(x, indices=y, k=4)
        result_values, result_indices = result[0], result[1]

        # Convert indices to match PyTorch's dtype
        result_indices = result_indices.to(torch.int64)

        # Compare results
        torch.testing.assert_close(result_values, ref_values)
        torch.testing.assert_close(result_indices, ref_indices)

    def test_topk_dynamic_dim(self):
        dtype = torch.float32
        # Test with different last dimension sizes
        for dim in [64, 128, 256]:
            x = torch.rand([8, dim], dtype=dtype)

            # Get reference result
            ref_values, ref_indices = torch.topk(x, k=4, dim=-1)

            # Get result from our kernel
            y = torch.arange(dim)[None, :].repeat(8, 1).to(torch.int32)
            result = kernels.iree_topk(x, indices=y, k=4)
            result_values, result_indices = result[0], result[1]

            # Convert indices to match PyTorch's dtype
            result_indices = result_indices.to(torch.int64)

            # Compare results
            torch.testing.assert_close(
                result_values, ref_values, msg=f"Failed for dim={dim}"
            )
            torch.testing.assert_close(
                result_indices, ref_indices, msg=f"Failed for dim={dim}"
            )
