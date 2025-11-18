# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for flatten op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import flatten_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestFlatten(OpComparisonTestBase):
    """Test flatten implementations."""

    @parameterized.expand(
        [
            # Basic flatten tests
            ((2, 3, 4), 0, -1, torch.float32),
            ((2, 3, 4), 1, 2, torch.float32),
            ((2, 3, 4, 5), 1, 3, torch.float32),
            ((2, 3, 4, 5), 0, 2, torch.float32),
            # Edge cases
            ((10,), 0, 0, torch.float32),
            ((2, 3, 4), 2, 2, torch.float32),
            # Different dtypes
            ((2, 3, 4), 0, -1, torch.float16),
            ((2, 3, 4), 1, 2, torch.int32),
            # Test negative dimensions
            ((2, 3, 4, 5), -3, -1, torch.float32),
            ((2, 3, 4, 5), -2, -1, torch.float32),
            ((2, 3, 4), -1, -1, torch.float32),
        ]
    )
    def test_flatten_variants(self, input_shape, start_dim, end_dim, dtype):
        """Test flatten with various dimensions."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.flatten,
            reference_impl=flatten_default,
            test_impls="all",
            args=[input_tensor, start_dim, end_dim],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
