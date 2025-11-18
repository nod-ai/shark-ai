# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for permute op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import permute as permute_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestPermute(OpComparisonTestBase):
    """Test permute implementations."""

    @parameterized.expand(
        [
            # Basic permute tests
            ((2, 3, 4), [0, 2, 1], torch.float32),
            ((2, 3, 4), [2, 1, 0], torch.float32),
            ((2, 3, 4), [1, 0, 2], torch.float32),
            ((2, 3, 4, 5), [0, 2, 1, 3], torch.float32),
            ((2, 3, 4, 5), [3, 2, 1, 0], torch.float32),
            ((2, 3, 4, 5, 6), [4, 3, 2, 1, 0], torch.float32),
            ((2, 3, 4, 5, 6), [0, 4, 1, 3, 2], torch.float32),
            # Identity permute
            ((2, 3, 4), [0, 1, 2], torch.float32),
            # Different dtypes
            ((2, 3, 4), [0, 2, 1], torch.float16),
            ((2, 3, 4), [2, 1, 0], torch.int32),
        ]
    )
    def test_permute_variants(self, input_shape, dims, dtype):
        """Test permute with various dimension permutations."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.permute,
            reference_impl=permute_default,
            test_impls="all",
            args=[input_tensor, dims],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
