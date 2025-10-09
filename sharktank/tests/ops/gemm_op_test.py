# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for GEMM op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops.default_impls import gemm_default
from sharktank.utils.testing import (
    OpComparisonTestBase,
    OpTestConfig,
    assert_cosine_similarity_close,
    create_test_tensor,
)


class TestGemm(OpComparisonTestBase):
    """Test GEMM implementations."""

    @parameterized.expand(
        [
            # Basic GEMM tests with c matrix
            ((4, 8), (8, 16), (4, 16), 1.0, 1.0, False, False, torch.float32),
            ((4, 8), (8, 16), (4, 16), 2.0, 3.0, False, False, torch.float32),
            # With transpose_a
            ((8, 4), (8, 16), (4, 16), 1.0, 1.0, True, False, torch.float32),
            ((8, 4), (8, 16), (4, 16), 2.0, 3.0, True, False, torch.float32),
            # With transpose_b
            ((4, 8), (16, 8), (4, 16), 1.0, 1.0, False, True, torch.float32),
            ((4, 8), (16, 8), (4, 16), 2.0, 3.0, False, True, torch.float32),
            # With both transposes
            ((8, 4), (16, 8), (4, 16), 1.0, 1.0, True, True, torch.float32),
            ((8, 4), (16, 8), (4, 16), 2.0, 3.0, True, True, torch.float32),
            # Different dtypes
            ((4, 8), (8, 16), (4, 16), 1.0, 1.0, False, False, torch.float16),
            ((4, 8), (8, 16), (4, 16), 1.0, 1.0, False, False, torch.bfloat16),
            # Zero alpha or beta
            ((4, 8), (8, 16), (4, 16), 0.0, 1.0, False, False, torch.float32),
            ((4, 8), (8, 16), (4, 16), 1.0, 0.0, False, False, torch.float32),
            # Tensor scalars
            (
                (4, 8),
                (8, 16),
                (4, 16),
                torch.tensor(2.0),
                torch.tensor(3.0),
                False,
                False,
                torch.float32,
            ),
        ]
    )
    def test_gemm_variants(
        self, a_shape, b_shape, c_shape, alpha, beta, transa, transb, dtype
    ):
        """Test GEMM with various configurations."""
        torch.manual_seed(42)

        a = create_test_tensor(a_shape, dtype)
        b = create_test_tensor(b_shape, dtype)
        c = create_test_tensor(c_shape, dtype) if c_shape else None

        config = OpTestConfig(
            op=ops.gemm,
            reference_impl=gemm_default,
            test_impls="all",
            args=[a, b, c, alpha, beta, transa, transb],
            kwargs={},
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
        )
        self.compare_implementations(config)

    @parameterized.expand(
        [
            # Cases without c matrix (c=None) - many implementations return NotImplemented
            ((4, 8), (8, 16), None, 1.0, None, False, False, torch.float32),
            ((4, 8), (8, 16), None, 2.0, None, False, False, torch.float32),
        ]
    )
    def test_gemm_no_c_matrix(
        self, a_shape, b_shape, c_shape, alpha, beta, transa, transb, dtype
    ):
        """Test GEMM cases without c matrix where many implementations return NotImplemented."""
        torch.manual_seed(42)

        a = create_test_tensor(a_shape, dtype)
        b = create_test_tensor(b_shape, dtype)
        c = create_test_tensor(c_shape, dtype) if c_shape else None

        config = OpTestConfig(
            op=ops.gemm,
            reference_impl=gemm_default,
            test_impls="all",
            args=[a, b, c, alpha, beta, transa, transb],
            kwargs={},
            atol=0.05,  # Use cosine similarity tolerance
            comparison_fn=lambda ref, test, *, atol, **_: assert_cosine_similarity_close(
                test, ref, atol=atol
            ),
            fail_on_not_implemented=False,  # Skip implementations that return NotImplemented
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
