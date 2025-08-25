# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for transpose op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops.shaping import transpose
from sharktank.ops.shaping.transpose import transpose_default
from sharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestTranspose(OpComparisonTestBase):
    """Test transpose implementations."""

    @parameterized.expand(
        [
            # Basic transpose tests
            ((2, 3), 0, 1, torch.float32),
            ((2, 3, 4), 0, 2, torch.float32),
            ((2, 3, 4), 1, 2, torch.float32),
            ((2, 3, 4, 5), 0, 3, torch.float32),
            ((2, 3, 4, 5), 1, 3, torch.float32),
            # Identity transpose
            ((3, 4), 0, 0, torch.float32),
            ((3, 4), 1, 1, torch.float32),
            # Different dtypes
            ((2, 3, 4), 0, 2, torch.float16),
            ((2, 3, 4), 1, 2, torch.int32),
            # Test negative dimensions
            ((2, 3, 4), -1, -2, torch.float32),
            ((2, 3, 4), 0, -1, torch.float32),
            ((2, 3, 4, 5), -2, -1, torch.float32),
        ]
    )
    def test_transpose_variants(self, input_shape, dim0, dim1, dtype):
        """Test transpose with various dimension pairs."""
        torch.manual_seed(42)

        # Create test tensor
        input_tensor = torch.randn(input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.transpose,
            reference_impl=transpose_default,
            test_impls="all",
            args=[input_tensor, dim0, dim1],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
