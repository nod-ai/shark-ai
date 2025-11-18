# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for squeeze op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import squeeze_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestSqueeze(OpComparisonTestBase):
    """Test squeeze implementations."""

    @parameterized.expand(
        [
            # Squeeze all dimensions
            ((1, 3, 1, 4, 1), None, torch.float32),
            ((1, 1, 1), None, torch.float32),
            ((2, 1, 3), None, torch.float32),
            # Squeeze specific dimensions
            ((1, 3, 4), 0, torch.float32),
            ((2, 1, 4), 1, torch.float32),
            ((2, 3, 1), 2, torch.float32),
            ((1, 3, 1, 4), 0, torch.float32),
            ((1, 3, 1, 4), 2, torch.float32),
            # Different dtypes
            ((1, 3, 1, 4), 0, torch.float16),
            ((2, 1, 4), 1, torch.int32),
            # Test negative dimensions
            ((2, 3, 1, 4), -1, torch.float32),
            ((1, 3, 1, 4), -3, torch.float32),
            ((2, 1, 3, 1), -2, torch.float32),
        ]
    )
    def test_squeeze_variants(self, input_shape, dim, dtype):
        """Test squeeze with various dimensions."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.squeeze,
            reference_impl=squeeze_default,
            test_impls="all",
            args=[input_tensor, dim],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
