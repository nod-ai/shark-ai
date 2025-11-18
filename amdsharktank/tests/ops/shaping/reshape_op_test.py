# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for reshape op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import reshape_default
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestReshape(OpComparisonTestBase):
    """Test reshape implementations."""

    @parameterized.expand(
        [
            # Basic reshape tests
            ((2, 3, 4), (6, 4), torch.float32),
            ((2, 3, 4), (2, 12), torch.float32),
            ((8, 8), (4, 16), torch.float32),
            ((10, 1, 5), (10, 5), torch.float32),
            # Different dtypes
            ((2, 3, 4), (6, 4), torch.float16),
            ((2, 3, 4), (6, 4), torch.int32),
            # More complex reshapes
            ((2, 3, 4, 5), (2, 60), torch.float32),
            ((1, 1, 1, 10), (10,), torch.float32),
            ((6,), (2, 3), torch.float32),
            # Test dynamic dimensions (-1)
            ((6, 4), (-1, 4), torch.float32),
            ((6, 4), (6, -1), torch.float32),
            ((2, 3, 4), (-1, 12), torch.float32),
            ((24,), (2, -1, 4), torch.float32),
        ]
    )
    def test_reshape_variants(self, input_shape, output_shape, dtype):
        """Test reshape with various input and output shapes."""
        torch.manual_seed(42)

        # Create test tensor
        if dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)

        config = OpTestConfig(
            op=ops.reshape,
            reference_impl=reshape_default,
            test_impls="all",
            args=[input_tensor, list(output_shape)],
            kwargs={},
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
