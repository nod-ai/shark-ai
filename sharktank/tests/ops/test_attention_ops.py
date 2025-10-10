# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for attention op implementations."""

import unittest
import torch
from parameterized import parameterized

from sharktank import ops
from sharktank.ops import attention_impls
from sharktank.utils.testing import OpComparisonTestBase, OpTestConfig


class TestScaledDotProductAttention(OpComparisonTestBase):
    """Test scaled dot product attention implementations."""

    @parameterized.expand(
        [
            # No causal, no mask
            (2, 8, 128, 64, torch.float16, False, False, None, None),
            (2, 8, 128, 64, torch.float32, False, False, None, None),
            # Test causal attention
            (2, 8, 128, 64, torch.float16, True, False, None, None),
            (2, 8, 128, 64, torch.float16, True, False, 0.125, None),
            # Test explicit masking
            (2, 8, 128, 64, torch.float16, False, True, None, None),
            (2, 8, 256, 64, torch.float32, False, True, None, None),
            # Test softcap
            (1, 4, 64, 32, torch.float32, False, False, None, 50.0),
            # Test return_lse
            (2, 8, 128, 64, torch.float32, False, False, None, None),
        ]
    )
    def test_attention_variants(
        self,
        batch,
        heads,
        seq_len,
        head_dim,
        dtype,
        is_causal,
        has_mask,
        scale,
        softcap,
    ):
        """Test attention with various configurations."""
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)

        if has_mask:
            # Create a simple attention mask with shape [1, 1, seq_len, seq_len]
            # This broadcasts across all batches and heads
            mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            a = mask.to(dtype)
        else:
            a = None

        fail_on_not_implemented = softcap is None

        if dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 3e-2, 3e-2
        else:
            atol, rtol = 3e-3, 3e-3

        # Wrapper to extract output from tuple for OpTestConfig
        # OpTestConfig calls unbox_tensor before comparison_fn, so we must unwrap here
        def op_wrapper(q, k, v, a, **kwargs):
            out, lse = ops.scaled_dot_product_attention(q, k, v, a, **kwargs)
            # Verify LSE is None when return_lse=False
            assert lse is None, f"LSE should be None when return_lse=False, got {lse}"
            return out

        def reference_wrapper(q, k, v, a, **kwargs):
            out, lse = attention_impls.scaled_dot_product_attention_decomposed(
                q, k, v, a, **kwargs
            )
            # Verify LSE is None when return_lse=False
            assert (
                lse is None
            ), f"Reference LSE should be None when return_lse=False, got {lse}"
            return out

        # Use decomposed as reference since it supports all features
        config = OpTestConfig(
            op=op_wrapper,
            reference_impl=reference_wrapper,
            test_impls="all",
            args=[q, k, v, a],
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "softcap": softcap,
                "impl": None,
                "return_lse": False,
            },
            atol=atol,
            rtol=rtol,
            fail_on_not_implemented=fail_on_not_implemented,
        )
        self.compare_implementations(config)

    def test_attention_lse_output(self):
        """Test that LSE is computed correctly when requested."""
        torch.manual_seed(42)
        batch, heads, seq_len, head_dim = 2, 4, 64, 32
        dtype = torch.float32

        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        a = None

        # Test with return_lse=True (decomposed only, since others don't support it)
        out_with_lse, lse = ops.scaled_dot_product_attention(
            q,
            k,
            v,
            a,
            is_causal=False,
            scale=None,
            softcap=None,
            impl="decomposed",
            return_lse=True,
        )

        # Test with return_lse=False
        out_without_lse, lse_none = ops.scaled_dot_product_attention(
            q,
            k,
            v,
            a,
            is_causal=False,
            scale=None,
            softcap=None,
            impl="decomposed",
            return_lse=False,
        )

        # Outputs should be identical
        torch.testing.assert_close(out_with_lse, out_without_lse, atol=1e-5, rtol=1e-5)

        # LSE should have correct shape when requested
        self.assertIsNotNone(lse)
        self.assertEqual(lse.shape, (batch, heads, seq_len))

        # LSE should be None when not requested
        self.assertIsNone(lse_none)

        # Verify LSE is reasonable (all values should be finite)
        self.assertTrue(torch.all(torch.isfinite(lse)))


if __name__ == "__main__":
    unittest.main()
