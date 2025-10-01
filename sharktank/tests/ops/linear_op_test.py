# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for linear op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops.default_impls import linear_default
from sharktank.ops.qlinear_impls import qlinear_tensor_scaled, linear_quantized_weight
from sharktank.utils.testing import (
    OpComparisonTestBase,
    OpTestConfig,
    assert_cosine_similarity_close,
    create_test_tensor,
)


class TestLinear(OpComparisonTestBase):
    """Test linear implementations."""

    @parameterized.expand(
        [
            # Basic linear tests - using input dtype as accum_dtype
            ((2, 8), (16, 8), None, torch.float32),
            ((2, 8), (16, 8), (16,), torch.float32),
            # Batch dimensions
            ((4, 2, 8), (16, 8), None, torch.float32),
            ((4, 2, 8), (16, 8), (16,), torch.float32),
            ((2, 3, 4, 8), (16, 8), None, torch.float32),
            ((2, 3, 4, 8), (16, 8), (16,), torch.float32),
            # Different dtypes
            ((2, 8), (16, 8), None, torch.float16),
            ((2, 8), (16, 8), (16,), torch.float16),
            ((2, 8), (16, 8), None, torch.bfloat16),
            ((2, 8), (16, 8), (16,), torch.bfloat16),
            # Larger tensors
            ((32, 64), (128, 64), None, torch.float32),
            ((32, 64), (128, 64), (128,), torch.float32),
        ]
    )
    def test_linear_variants(self, input_shape, weight_shape, bias_shape, dtype):
        """Test linear with various shapes, bias configurations, and dtypes."""
        torch.manual_seed(42)

        input_tensor = create_test_tensor(input_shape, dtype)
        weight = create_test_tensor(weight_shape, dtype)
        bias = create_test_tensor(bias_shape, dtype) if bias_shape else None

        kwargs = {
            "matmul_impl": "*",
            "accum_dtype": None,
        }

        config = OpTestConfig(
            op=ops.linear,
            reference_impl=linear_default,
            test_impls="all",
            skip_impls=[
                qlinear_tensor_scaled,  # Doesn't handle float32 quantized tensors - should but doesn't
            ],
            args=[input_tensor, weight, bias],
            kwargs=kwargs,
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
        )
        self.compare_implementations(config)

    @parameterized.expand(
        [
            # Accumulation dtype tests - different from input dtype
            ((2, 8), (16, 8), (16,), torch.float16, torch.float32),
            ((2, 8), (16, 8), (16,), torch.bfloat16, torch.bfloat16),
        ]
    )
    def test_linear_accum_dtype(
        self, input_shape, weight_shape, bias_shape, dtype, accum_dtype
    ):
        """Test linear with explicit accumulation dtypes different from input dtype."""
        torch.manual_seed(42)

        input_tensor = create_test_tensor(input_shape, dtype)
        weight = create_test_tensor(weight_shape, dtype)
        bias = create_test_tensor(bias_shape, dtype) if bias_shape else None

        kwargs = {
            "matmul_impl": "*",
            "accum_dtype": accum_dtype,
        }

        config = OpTestConfig(
            op=ops.linear,
            reference_impl=linear_default,
            test_impls="all",
            skip_impls=[
                linear_quantized_weight,  # Doesn't support explicit accum_dtype
            ],
            args=[input_tensor, weight, bias],
            kwargs=kwargs,
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
            fail_on_not_implemented=False,  # Skip implementations that don't support different accum_dtype
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
