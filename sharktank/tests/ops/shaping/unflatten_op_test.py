# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for unflatten op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import unflatten_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestUnflatten(OpComparisonTestBase):
    """Test unflatten implementations."""

    @parameterized.expand(
        [
            # Basic unflatten tests
            ((6, 4), 0, (2, 3), torch.float32),
            ((2, 12), 1, (3, 4), torch.float32),
            ((24,), 0, (2, 3, 4), torch.float32),
            ((8, 8), 1, (2, 4), torch.float32),
            # Different dtypes
            ((6, 4), 0, (2, 3), torch.float16),
            ((2, 12), 1, (3, 4), torch.int32),
            # Edge cases
            ((10,), 0, (10,), torch.float32),
            ((6, 4), 1, (4,), torch.float32),
            # Test negative dimensions
            ((6, 4), -2, (2, 3), torch.float32),
            ((2, 12), -1, (3, 4), torch.float32),
            ((24,), -1, (2, 3, 4), torch.float32),
        ]
    )
    def test_unflatten_variants(self, input_shape, dim, sizes, dtype):
        """Test unflatten with various dimensions and sizes."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.unflatten,
            reference_impl=unflatten_default,
            test_impls="all",
            args=[input_tensor, dim, sizes],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
