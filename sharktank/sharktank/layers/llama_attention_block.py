# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import math

import torch
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .rotary_embedding import RotaryEmbeddingLayer

__all__ = [
    "LlamaAttentionBlock",
]


class LlamaAttentionBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama."""

    def __init__(
        self,
        theta: Theta,
        *,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
    ):
        super().__init__(theta)
        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("attn_q", LinearLayer(theta("attn_q")))
        self.add_module("attn_k", LinearLayer(theta("attn_k")))
        self.add_module("attn_v", LinearLayer(theta("attn_v")))
        self.add_module("attn_output", LinearLayer(theta("attn_output")))

        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv

    def forward(
        self,
        h: torch.Tensor,
        *,
        embedding: RotaryEmbeddingLayer,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        start_index: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.attn_norm(h)

        bs, q_len, feature_dim = x.shape
        kv_seq_len = start_index + q_len
        assert feature_dim == self.head_count * self.head_dim

        xq = self.attn_q(x)
        xk = self.attn_k(x)
        xv = self.attn_v(x)

        xq = xq.view(bs, q_len, self.head_count, self.head_dim)
        xk = xk.view(bs, q_len, self.head_count_kv, self.head_dim)
        xv = xv.view(bs, q_len, self.head_count_kv, self.head_dim)

        # Fast path to start_index based embedding lookup if available.
        # Falls back to a slower position based index lookup.
        if start_index is not None:
            xq = embedding.forward(xt=xq, start_index=start_index)
            xk = embedding.forward(xt=xk, start_index=start_index)
        else:
            xq = embedding.apply_batched_mask(xt=xq, mask=embedding_batch_mask)
            xk = embedding.apply_batched_mask(xt=xk, mask=embedding_batch_mask)

        # Expand kv heads for GQA.
        gqa_n_rep = self.head_count // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:

            def repeat_kv(x: torch.Tensor) -> torch.Tensor:
                bs, slen, n_kv_heads, head_dim = x.shape
                return (
                    x.unsqueeze(-2)
                    .expand(bs, slen, n_kv_heads, gqa_n_rep, head_dim)
                    .reshape(bs, slen, n_kv_heads * gqa_n_rep, head_dim)
                )

            xk = repeat_kv(xk)
            xv = repeat_kv(xv)

        # Update our positions in the cache.
        cache_k[:bs, start_index:kv_seq_len] = xk
        cache_v[:bs, start_index:kv_seq_len] = xv

        # Derive keys/values from the entirety of the available sequence.
        keys = cache_k[:bs, :kv_seq_len]
        values = cache_v[:bs, :kv_seq_len]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Flash attention.
        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask.
        if attention_mask is not None:
            expected_mask_shape = (bs, 1, q_len, kv_seq_len)
            assert (
                attention_mask.shape == expected_mask_shape
            ), f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.shape}"
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        attn_output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bs, q_len, -1)

        # Project.
        attn_output = self.attn_output(attn_output)

        # Remainder of the block.
        h = h + attn_output

        return h
