# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for view op implementations."""

import unittest
import torch
from parameterized import parameterized

from amdsharktank import ops
from amdsharktank.ops.default_impls import (
    view_default,
    view_block_scaled_i4_layout,
    view_quantized_dispatcher,
)
from amdsharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestView(OpComparisonTestBase):
    """Test view implementations."""

    @parameterized.expand(
        [
            # Basic view tests with shape
            ((2, 3, 4), (6, 4), None, torch.float32),
            ((2, 3, 4), (2, 12), None, torch.float32),
            ((8, 8), (4, 16), None, torch.float32),
            ((10, 1, 5), (10, 5), None, torch.float32),
            # Different dtypes
            ((2, 3, 4), (6, 4), None, torch.float16),
            ((2, 3, 4), (6, 4), None, torch.int32),
            # Test dynamic dimensions (-1)
            ((6, 4), (-1, 4), None, torch.float32),
            ((6, 4), (6, -1), None, torch.float32),
            ((2, 3, 4), (-1, 12), None, torch.float32),
            ((24,), (2, -1, 4), None, torch.float32),
        ]
    )
    def test_view_variants(self, input_shape, output_shape, target_dtype, input_dtype):
        """Test view with various input shapes, output shapes, and dtypes."""
        torch.manual_seed(42)

        # Create test tensor
        if input_dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=input_dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=input_dtype)

        config = OpTestConfig(
            op=ops.view,
            reference_impl=view_default,
            test_impls="all",
            skip_impls=[view_block_scaled_i4_layout],
            args=[input_tensor],
            kwargs={"shape": output_shape, "dtype": target_dtype},
        )
        self.compare_implementations(config)

    # Type conversion tests
    @parameterized.expand(
        [
            ((2, 3, 4), (2, 3, 4), None, torch.int32),
            ((2, 3, 4), None, torch.int32, torch.float32),
        ]
    )
    def test_view_variants_not_implemented(
        self, input_shape, output_shape, target_dtype, input_dtype
    ):
        """Test view with various input shapes, output shapes, and dtypes."""
        torch.manual_seed(42)

        # Create test tensor
        if input_dtype.is_floating_point:
            input_tensor = torch.randn(input_shape, dtype=input_dtype)
        else:
            input_tensor = torch.randint(0, 10, input_shape, dtype=input_dtype)

        config = OpTestConfig(
            op=ops.view,
            reference_impl=view_default,
            test_impls="all",
            skip_impls=[view_block_scaled_i4_layout, view_quantized_dispatcher],
            args=[input_tensor],
            kwargs={"shape": output_shape, "dtype": target_dtype},
            fail_on_not_implemented=False,
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
