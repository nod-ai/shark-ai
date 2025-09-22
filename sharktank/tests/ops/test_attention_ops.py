# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for attention op implementations."""

import unittest
import torch
from parameterized import parameterized
from typing import Optional
from sharktank import ops
from sharktank.ops import attention_impls
from sharktank.utils.testing import OpComparisonTestBase, OpTestConfig


def build_mask(
    mask: Optional[torch.Tensor],
    sliding_window: Optional[int],
    kv_size: int,
    n_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
):
    if sliding_window is None or sliding_window <= 0:
        if mask is None:
            mask = torch.full(
                (n_tokens, n_tokens),
                float("-inf"),
                dtype=dtype,
                device=device,
            )
            mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        return mask.to(device)

    is_prefill = kv_size == n_tokens
    if is_prefill:
        # prefill path: causal mask within sliding window
        if mask is None:
            mask = torch.triu(
                torch.full(
                    (n_tokens, n_tokens), -float("inf"), dtype=dtype, device=device
                ),
                diagonal=1,
            )

        if sliding_window > 0:
            sliding_window_mask = torch.tril(
                torch.full(
                    (n_tokens, n_tokens), -float("inf"), dtype=dtype, device=device
                ),
                diagonal=-sliding_window,
            )
            mask = mask.to(device) + sliding_window_mask

    else:
        # decode path
        if sliding_window > 0 and kv_size > sliding_window:
            start_idx = kv_size - sliding_window
            neg_inf = float("-inf")
            mask[..., :start_idx] = neg_inf

    return mask.to(device)


class TestScaledDotProductAttention(OpComparisonTestBase):
    """Test scaled dot product attention implementations."""

    @parameterized.expand(
        [
            # No causal, no mask
            (2, 8, 128, 64, torch.float16, False, False, None, None, None, None),
            (2, 8, 128, 64, torch.float32, False, False, None, None, None, None),
            # Test causal attention
            (2, 8, 128, 64, torch.float16, True, False, None, None, None, None),
            (2, 8, 128, 64, torch.float16, True, False, 0.125, None, None, None),
            # Test explicit masking
            (2, 8, 128, 64, torch.float16, False, True, None, None, None, None),
            (2, 8, 256, 64, torch.float32, False, True, None, None, None, None),
            # Test softcap
            (1, 4, 64, 32, torch.float32, False, False, None, 50.0, None, None),
            # Test Sliding Window
            (2, 8, 128, 64, torch.float16, False, False, None, None, None, 32),
            (2, 8, 256, 64, torch.float32, False, False, None, None, None, 64),
            (1, 4, 64, 32, torch.bfloat16, False, False, None, None, None, 16),
            # Test Sink and Sliding Window
            (2, 8, 128, 64, torch.bfloat16, True, False, None, None, 0.25, 19),
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
        sink_scale,
        sliding_window,
    ):
        """Test attention with various configurations."""
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype)

        if has_mask:
            # Create a simple explicit attention mask
            mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            a = mask.to(dtype)
        else:
            # Use build_mask for all other cases (causal, sliding window, or none)
            # build_mask will handle the logic based on is_causal and sliding_window
            if is_causal or sliding_window is not None:
                a = build_mask(
                    None, sliding_window, k.shape[-2], q.shape[-2], dtype, q.device
                )
            else:
                a = None

        unsupported = (softcap is not None) or (sink_scale is not None)
        fail_on_not_implemented = not unsupported

        sink = (
            torch.full((1, heads), sink_scale, dtype=q.dtype)
            if sink_scale is not None
            else None
        )

        if dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 3e-2, 3e-2
        else:
            atol, rtol = 3e-3, 3e-3
        # Use decomposed as reference since it supports all features
        config = OpTestConfig(
            op=ops.scaled_dot_product_attention,
            reference_impl=attention_impls.scaled_dot_product_attention_decomposed,
            test_impls="all",
            args=[q, k, v, a],
            kwargs={
                "is_causal": is_causal,
                "scale": scale,
                "softcap": softcap,
                "impl": None,
                "sink": sink,
            },
            atol=atol,
            rtol=rtol,
            fail_on_not_implemented=fail_on_not_implemented,
        )
        self.compare_implementations(config)


if __name__ == "__main__":
    unittest.main()
