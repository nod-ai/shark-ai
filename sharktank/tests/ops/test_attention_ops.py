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
from sharktank.layers.paged_attention import PagedMHAttention
import math


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
            # Test explicit masking (full causal mask passed explicitly)
            (2, 8, 128, 64, torch.float16, False, True, None, None),
            (2, 8, 256, 64, torch.float32, False, True, None, None),
            # Test softcap (no causal, no mask)
            (1, 4, 64, 32, torch.float32, False, False, None, 50.0),
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
            # Explicit full causal mask (no sliding window) for regression against torch impls
            mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            a = mask.unsqueeze(0).unsqueeze(0).to(dtype)

        else:
            a = None

        unsupported = softcap is not None
        fail_on_not_implemented = not unsupported

        if dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 3e-2, 3e-2
        else:
            atol, rtol = 3e-3, 3e-3
        # Use decomposed as reference since it supports all features
        config = OpTestConfig(
            op=ops.scaled_dot_product_attention,
            reference_impl=attention_impls.scaled_dot_product_attention_decomposed,
            test_impls="all",
            # Provide placeholder sink=None as positional argument expected by reference impl
            args=[q, k, v, a, None],
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "softcap": softcap,
                "impl": None,
            },
            atol=atol,
            rtol=rtol,
            fail_on_not_implemented=fail_on_not_implemented,
        )
        self.compare_implementations(config)


class TestSlidingWindowMaskGolden(unittest.TestCase):
    def test_causal_mask(self):

        mask = PagedMHAttention.build_mask(
            PagedMHAttention,
            None,
            None,
            kv_size=4,
            n_tokens=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Each query can only see keys up to its own position
        expected_finite_keys = [
            [0],  # query 0 sees key 0
            [0, 1],  # query 1 sees keys 0,1
            [0, 1, 2],  # query 2 sees keys 0,1,2
            [0, 1, 2, 3],  # query 3 sees keys 0,1,2,3
        ]

        self._check_mask_pattern(mask, expected_finite_keys)

    def test_sliding_window_mask(self):
        """Test sliding window masking where kv_size > sliding_window."""
        mask = PagedMHAttention.build_mask(
            PagedMHAttention,
            None,
            sliding_window=2,
            kv_size=5,
            n_tokens=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Each query sees a sliding window of 2 keys around its position
        # kv_size=5 > sliding_window=2, so window is constrained by sliding_window
        expected_finite_keys = [
            [0],  # query 0: window starts at 0
            [0, 1],  # query 1: window covers 0,1
            [1, 2],  # query 2: window covers 1,2
            [2, 3],  # query 3: window covers 2,3
            [3, 4],  # query 4: window covers 3,4
        ]

        self._check_mask_pattern(mask, expected_finite_keys)

    def test_sliding_window_larger_than_kv(self):

        mask = PagedMHAttention.build_mask(
            PagedMHAttention,
            None,
            sliding_window=6,
            kv_size=4,
            n_tokens=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # When sliding_window > kv_size, window is effectively unconstrained by sliding_window
        # Should behave like causal masking since window is larger than sequence
        expected_finite_keys = [
            [0],  # query 0 sees key 0
            [0, 1],  # query 1 sees keys 0,1
            [0, 1, 2],  # query 2 sees keys 0,1,2
            [0, 1, 2, 3],  # query 3 sees keys 0,1,2,3
        ]

        self._check_mask_pattern(mask, expected_finite_keys)

    def _check_mask_pattern(self, mask, expected_finite_keys):
        for qi, expected_keys in enumerate(expected_finite_keys):
            row = mask[qi]
            finite_keys = (row > float("-inf")).nonzero(as_tuple=True)[0].tolist()
            self.assertEqual(
                finite_keys,
                expected_keys,
                f"Query {qi}: expected keys {expected_keys}, got {finite_keys}",
            )


class TestSinkAttentionGolden(unittest.TestCase):
    def test_sink_vs_no_sink_difference(self):
        torch.manual_seed(42)

        # 4D tensors: (batch=1, n_heads=1, n_tokens=2, head_dim=2)
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

        # Regular causal attention
        regular_result = ops.scaled_dot_product_attention(q, k, v, None, is_causal=True)

        # Sink attention
        sink = torch.tensor([0.5])
        sink_result = ops.scaled_dot_product_attention(
            q, k, v, None, is_causal=True, sink=sink
        )

        # Results should be different when sink is applied
        self.assertFalse(torch.allclose(regular_result, sink_result, atol=1e-6))

        # Both should have same shape
        self.assertEqual(regular_result.shape, sink_result.shape)

    def test_sink_softmax_behavior(self):
        """
        SINK ATTENTION DIFFERENCE:
        - Regular: softmax(attn_weights) -> each row sums to 1.0
        - Sink: softmax(cat([attn_weights, sink_value])) then slice off sink portion
               -> each row sums to LESS than 1.0 (sink absorbed probability mass)
        VISUAL EXAMPLE with 2x2 attention matrix:
         Regular attention weights after softmax:
           Query 0: [1.0, 0.0]  <- causal mask hides key 1
           Query 1: [0.33, 0.67] <- can see both keys, sums to 1.0

         Sink attention (with sink=0.5):
           Step 1: Concat sink -> [[weights, 0.5], [weights, 0.5]]
           Step 2: Softmax entire matrix -> normalizes including sink
           Step 3: Slice off sink column -> weights now sum < 1.0
              Query 0: [0.55, 0.0]  <- less than 1.0! sink absorbed 0.45
              Query 1: [0.21, 0.43] <- less than 1.0! sink absorbed ~0.36

            Simple explicit calculation (Query 0 only):
              Scaled logits (key0, key1(masked), sink) = [0.7071, -inf, 0.5]
              exp = [exp(0.7071)=2.0281, exp(-inf)=0, exp(0.5)=1.6487]
              Denominator = 2.0281 + 0 + 1.6487 = 3.6768
              Softmax (before slicing) = [2.0281/3.6768=0.5511, 0, 1.6487/3.6768=0.4489]
              After slicing off sink column -> retained = [0.5511, 0.0]; sink absorbed 0.4489 (~0.45)
        """

        # 4D tensors: (batch=1, n_heads=1, n_tokens=2, head_dim=2)
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        sink = torch.tensor([0.5], dtype=torch.float32)

        # Extract tensor dimensions
        bs, n_heads, n_tokens, head_dim = q.shape

        # Manual computation to verify sink softmax behavior
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.full((n_tokens, n_tokens), float("-inf")), diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        # Now test the key difference in sink attention:
        # Regular softmax (without sink)
        regular_weights = torch.softmax(attn_weights, dim=-1)

        # Sink weight
        sink_expanded = sink.reshape(1, 1, 1, 1).expand(1, 1, 2, 1)
        attn_with_sink = torch.cat([attn_weights, sink_expanded], dim=-1)
        sink_weights_full = torch.softmax(attn_with_sink, dim=-1)
        sink_weights = sink_weights_full[..., :-1]  # slice off sink portion

        # Verify shapes
        self.assertEqual(regular_weights.shape, (1, 1, 2, 2))
        self.assertEqual(sink_weights.shape, (1, 1, 2, 2))

        # The sink softmax should produce different attention weights
        self.assertFalse(torch.allclose(regular_weights, sink_weights, atol=1e-6))

        # Regular: With causal mask, query 0 can only see key 0, so gets weight 1.0
        # Sink: The sink value competes in softmax, reducing the weight for key 0
        # NOTE on tensor shape: attention weights are 4D = (batch, head, query_index, key_index).
        # In this tiny example batch=head=1, so we index as [0, 0, q, k] to pick the scalar for (query q -> key k).
        query_0_key_0_regular = regular_weights[0, 0, 0, 0].item()  # Query 0 -> Key 0
        query_0_key_1_regular = regular_weights[
            0, 0, 0, 1
        ].item()  # Query 0 -> Key 1 (masked)

        query_0_key_0_sink = sink_weights[
            0, 0, 0, 0
        ].item()  # Query 0 -> Key 0 with sink
        query_0_key_1_sink = sink_weights[
            0, 0, 0, 1
        ].item()  # Query 0 -> Key 1 (still masked)

        # Regular: query 0 gives full attention (1.0) to key 0, zero to masked key 1
        self.assertAlmostEqual(query_0_key_0_regular, 1.0, places=6)
        self.assertAlmostEqual(query_0_key_1_regular, 0.0, places=6)

        # Sink: query 0 gives LESS attention to key 0 (sink absorbed some probability)
        self.assertLess(
            query_0_key_0_sink, 1.0
        )  # Less than 1.0 due to sink competition
        self.assertAlmostEqual(
            query_0_key_1_sink, 0.0, places=6
        )  # Still masked (causal)

        # Check sdpa with explicit causal mask
        mask = torch.triu(torch.full((n_tokens, n_tokens), float("-inf")), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        production_with_mask = ops.scaled_dot_product_attention(
            q, k, v, mask, is_causal=False, sink=sink
        )
        expected_result = torch.matmul(sink_weights, v)
        self.assertTrue(
            torch.allclose(production_with_mask, expected_result, atol=1e-5)
        )


def test__invoke_golden_mask_cases():
    """Bridge test so pytest can invoke golden mask checks explicitly."""
    g = TestSlidingWindowMaskGolden()
    g.test_causal_mask()
    g.test_sliding_window_mask()
    g.test_sliding_window_larger_than_kv()

    s = TestSinkAttentionGolden()
    s.test_sink_vs_no_sink_difference()
    s.test_sink_softmax_behavior()


if __name__ == "__main__":
    unittest.main()
