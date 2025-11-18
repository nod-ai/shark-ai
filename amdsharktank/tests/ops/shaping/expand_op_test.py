# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for expand op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import expand_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestExpand(OpComparisonTestBase):
    """Test expand implementations."""

    @parameterized.expand(
        [
            # Basic expand tests
            ((1, 3, 1), [2, 3, 4], torch.float32),
            ((2, 1, 4), [2, 3, 4], torch.float32),
            ((1, 1, 4), [2, 3, 4], torch.float32),
            ((1,), [2, 3, 4], torch.float32),
            # Identity expand
            ((2, 3, 4), [2, 3, 4], torch.float32),
            # Broadcast leading dimensions
            ((3, 4), [2, 3, 4], torch.float32),
            ((4,), [2, 3, 4], torch.float32),
            # Different dtypes
            ((1, 3, 1), [2, 3, 4], torch.float16),
            ((2, 1, 4), [2, 3, 4], torch.int32),
            # Test with -1
            ((2, 3, 1), [2, -1, 4], torch.float32),
            ((1, 3, 4), [-1, 3, 4], torch.float32),
            ((2, 1, 4), [2, 5, -1], torch.float32),
        ]
    )
    def test_expand_variants(self, input_shape, expand_shape, dtype):
        """Test expand with various shapes."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.expand,
            reference_impl=expand_default,
            test_impls="all",
            args=[input_tensor, expand_shape],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
