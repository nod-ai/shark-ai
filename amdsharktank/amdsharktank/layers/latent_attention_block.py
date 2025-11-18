# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from amdsharktank.layers import CachedRotaryLayer
from amdsharktank.types import *
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from amdsharktank import ops

__all__ = [
    "LatentAttentionBlock",
]


class LatentAttentionBlock(ThetaLayer):
    """Implements a latent attention layer"""

    def __init__(
        self,
        theta: Theta,
        rms_epsilon: float,
        head_count: int,
        head_count_kv: int,
        rope_dimension_count: int,
        fake_quant: bool = False,
    ):
        super().__init__(theta)
        self.head_count = head_count
        self.head_count_kv = head_count_kv
        self.rope_dimension_count = rope_dimension_count

        self.add_module(
            "kv_norm", RMSNormLayer(theta("attn_kv_a_norm"), epsilon=rms_epsilon)
        )
        if "q" in theta:
            self.wq = LinearLayer(theta("q"), fake_quant=fake_quant)
        else:
            self.wq = None
            self.wq_a = LinearLayer(theta("attn_q_a"), fake_quant=fake_quant)
            self.wq_b = LinearLayer(theta("attn_q_b"), fake_quant=fake_quant)
            self.q_norm = RMSNormLayer(theta("attn_q_a_norm"), epsilon=rms_epsilon)

        self.add_module(
            "wkv_a", LinearLayer(theta("attn_kv_a_mqa"), fake_quant=fake_quant)
        )
        self.add_module("wkv_b", LinearLayer(theta("attn_kv_b"), fake_quant=fake_quant))

    def forward(
        self,
        h: torch.Tensor | ShardedTensor,
        embedding: CachedRotaryLayer,
        start_positions: InferenceTensor | None,
    ):
        if self.wq is not None:
            q = self.wq(h).unflatten(2, (self.head_count, -1))
        else:
            # (n_batches, seq_len, n_heads * (qk_nope_head_dim + qk_rope_head_dim))
            q = self.wq_b(self.q_norm(self.wq_a(h)))
            if isinstance(q, UnreducedTensor):
                q = ops.reduce_scatter(q, scatter_dim=2)
            q = q.unflatten(2, (self.head_count, -1))

        qk_nope_head_dim = q.shape[-1] - self.rope_dimension_count
        q_nope = q[:, :, :, :qk_nope_head_dim]
        q_rope = q[:, :, :, qk_nope_head_dim:]

        kv = self.wkv_a(h)
        kv_nope_size = kv.shape[-1] - self.rope_dimension_count
        if isinstance(kv, SplitPrimitiveTensor):
            kv = ops.replicate(kv, count=h.shard_count)
        kv_nope = kv[:, :, :kv_nope_size]
        k_rope = kv[:, :, kv_nope_size:]

        q_rope = embedding(xt=q_rope, start_positions=start_positions)
        k_rope = embedding(xt=k_rope.unsqueeze(2), start_positions=start_positions)

        xq = ops.cat((q_nope, q_rope), dim=-1)

        ##TODO: Restructure this to apply the wkv_b post attention instead of here
        kv_norm = self.kv_norm(kv_nope)
        # (n_batches, seq_len, n_heads * (v_head_dim + qk_nope_head_dim))
        wkv_b = self.wkv_b(kv_norm)
        wkv_b = wkv_b.unflatten(2, (self.head_count_kv, -1))

        k_nope = wkv_b[:, :, :, :qk_nope_head_dim]
        xv = wkv_b[:, :, :, qk_nope_head_dim:]

        k_rope = ops.repeat(k_rope, (1, 1, k_nope.shape[2] // k_rope.shape[2], 1))
        if isinstance(k_rope, ShardedTensor):
            k_rope = ops.reshard_like(k_rope, like=k_nope)

        xk = ops.cat((k_nope, k_rope), dim=-1)

        return xq, xk, xv
