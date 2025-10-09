# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for matmul op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops.default_impls import matmul_default
from sharktank.ops.custom_impls import (
    matmul_generic_tensor_block_scaled,
    matmul_generic_tensor_block_scaled_i4,
    matmul_generic_tensor_block_scaled_fp4_iree,
    matmul_generic_tensor_super_block_offset_scaled_4_6_i4,
)
from sharktank.utils.testing import (
    OpComparisonTestBase,
    OpTestConfig,
    assert_cosine_similarity_close,
    create_test_tensor,
)


class TestMatmul(OpComparisonTestBase):
    """Test matmul implementations."""

    @parameterized.expand(
        [
            # Basic matmul tests - dimensions must be multiples of 32 for quantized variants
            ((4, 32), (64, 32), True, torch.float32, torch.float32),
            # Batch dimensions
            ((2, 4, 32), (64, 32), True, torch.float32, torch.float32),
            ((2, 3, 4, 32), (64, 32), True, torch.float32, torch.float32),
            # Different dtypes
            ((4, 32), (64, 32), True, torch.float16, torch.float16),
            ((4, 32), (64, 32), True, torch.bfloat16, torch.bfloat16),
            # Mixed dtypes
            ((4, 32), (64, 32), True, torch.float32, torch.float16),
            # Larger tensors
            ((32, 64), (128, 64), True, torch.float32, torch.float32),
        ]
    )
    def test_matmul_variants(
        self, lhs_shape, rhs_shape, transpose_rhs, lhs_dtype, rhs_dtype
    ):
        """Test matmul with various shapes, transpose options, and dtypes."""
        torch.manual_seed(42)

        lhs = create_test_tensor(lhs_shape, lhs_dtype)
        rhs = create_test_tensor(rhs_shape, rhs_dtype)

        config = OpTestConfig(
            op=ops.matmul,
            reference_impl=matmul_default,
            test_impls="all",
            skip_impls=[
                matmul_generic_tensor_block_scaled_fp4_iree,  # Doesn't handle batches
                matmul_generic_tensor_block_scaled,  # Incorrectly implemented; does not work for BlockScaledLayouts
                matmul_generic_tensor_block_scaled_i4,
                matmul_generic_tensor_super_block_offset_scaled_4_6_i4,
            ],  # Integer types don't have quantizers suitable for testing
            args=[lhs, rhs],
            kwargs={"transpose_rhs": transpose_rhs},
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
        )
        self.compare_implementations(config)

    @parameterized.expand(
        [
            # Test cases where most implementations return NotImplemented (transpose_rhs=False)
            ((4, 32), (32, 64), False, torch.float32, torch.float32),
            ((4, 32), (32, 64), False, torch.float16, torch.float16),
            ((2, 4, 32), (32, 64), False, torch.float32, torch.float32),
        ]
    )
    def test_matmul_not_implemented_cases(
        self, lhs_shape, rhs_shape, transpose_rhs, lhs_dtype, rhs_dtype
    ):
        """Test matmul cases where many implementations return NotImplemented."""
        torch.manual_seed(42)

        lhs = create_test_tensor(lhs_shape, lhs_dtype)
        rhs = create_test_tensor(rhs_shape, rhs_dtype)

        config = OpTestConfig(
            op=ops.matmul,
            reference_impl=matmul_default,
            test_impls="all",
            skip_impls=[
                matmul_generic_tensor_block_scaled_fp4_iree,  # Doesn't handle batches
                matmul_generic_tensor_block_scaled,  # Incorrectly implemented; does not work for BlockScaledLayouts
                matmul_generic_tensor_block_scaled_i4,
                matmul_generic_tensor_super_block_offset_scaled_4_6_i4,
            ],  # Integer types don't have quantizers suitable for testing
            args=[lhs, rhs],
            kwargs={"transpose_rhs": transpose_rhs},
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
            fail_on_not_implemented=False,  # Skip implementations that return NotImplemented
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
