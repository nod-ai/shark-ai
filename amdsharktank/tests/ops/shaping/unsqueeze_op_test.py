# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for unsqueeze op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import unsqueeze_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestUnsqueeze(OpComparisonTestBase):
    """Test unsqueeze implementations."""

    @parameterized.expand(
        [
            # Basic unsqueeze tests
            ((2, 3, 4), 0, torch.float32),
            ((2, 3, 4), 1, torch.float32),
            ((2, 3, 4), 2, torch.float32),
            ((2, 3, 4), 3, torch.float32),
            # Edge cases
            ((5,), 0, torch.float32),
            ((5,), 1, torch.float32),
            # Different dtypes
            ((2, 3, 4), 0, torch.float16),
            ((2, 3, 4), 1, torch.int32),
            # Test negative dimensions
            ((2, 3, 4), -1, torch.float32),
            ((2, 3, 4), -2, torch.float32),
            ((2, 3, 4), -3, torch.float32),
            ((2, 3, 4), -4, torch.float32),
            ((5,), -1, torch.float32),
            ((5,), -2, torch.float32),
        ]
    )
    def test_unsqueeze_variants(self, input_shape, dim, dtype):
        """Test unsqueeze with various dimensions."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.unsqueeze,
            reference_impl=unsqueeze_default,
            test_impls="all",
            args=[input_tensor, dim],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
