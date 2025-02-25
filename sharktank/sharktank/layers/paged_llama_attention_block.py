# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import math

import torch
import torch.nn.functional as F
from ..types import QuantizerTensor
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .rotary_embedding import RotaryEmbeddingLayer
from .kv_cache import PagedKVCache
from .. import ops

__all__ = [
    "PagedLlamaAttentionBlock",
]


class PagedLlamaAttentionBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama using a
    paged cache."""

    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        cache: PagedKVCache,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        attention_dtype: Optional[torch.dtype] = None,
        attention_kernel: str = "decomposed",
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
    ):
        super().__init__(theta)

        self.block_index = block_index
        self.cache = cache
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_dtype = attention_dtype
        self.attention_kernel = attention_kernel
        self.attention_scale = attention_scale
        self.softcap = softcap
        self.fake_quant = fake_quant

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module(
            "attn_q", LinearLayer(theta("attn_q"), fake_quant=self.fake_quant)
        )
        self.add_module(
            "attn_k", LinearLayer(theta("attn_k"), fake_quant=self.fake_quant)
        )
        self.add_module(
            "attn_v", LinearLayer(theta("attn_v"), fake_quant=self.fake_quant)
        )
        self.add_module(
            "attn_output", LinearLayer(theta("attn_output"), fake_quant=self.fake_quant)
        )
        self.cache_quantizer = None
        if "kv_cache" in theta.keys:
            self.cache_quantizer: Optional[QuantizerTensor] = theta.optional_tensor(
                "kv_cache.quantizer"
            )

        if theta.optional_tensor("attn_output_norm") is None:
            self.add_module(
                "attn_output_norm",
                torch.nn.Identity(),
            )
        else:
            self.add_module(
                "attn_output_norm",
                RMSNormLayer(theta("attn_output_norm"), epsilon=rms_epsilon),
            )

    def forward(
        self,
        h: torch.Tensor,
        *,
        embedding: RotaryEmbeddingLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
    ):
        assert bool(start_index is not None) ^ bool(embedding_batch_mask is not None)
        x = self.attn_norm(h)
        bs, batch_seq_len, _ = x.shape

        xq = self.attn_q(x)
        xk = self.attn_k(x)
        xv = self.attn_v(x)

        assert xq.shape[-1] == self.head_count * self.head_dim
        assert xk.shape[-1] == self.head_count_kv * self.head_dim
        assert xv.shape[-1] == self.head_count_kv * self.head_dim

        xq = xq.view(bs, batch_seq_len, self.head_count, self.head_dim)
        xk = xk.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)
        xv = xv.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)

        # Fast path to start_index based embedding lookup if available.
        # Falls back to a slower position based index lookup.
        if start_index is not None:
            xq = embedding.forward(xt=xq, start_index=start_index)
            xk = embedding.forward(xt=xk, start_index=start_index)
        else:
            xq = embedding.apply_batched_mask(xt=xq, mask=embedding_batch_mask)
            xk = embedding.apply_batched_mask(xt=xk, mask=embedding_batch_mask)

        # Full sequence length.
        kv_seq_len = seq_block_ids.shape[1] * self.cache.block_seq_stride

        # Used by fp8_e4m3fnuz model
        if self.cache_quantizer is not None:
            if not self.fake_quant:
                # TODO: this seems like a bastardization of our quantized tensor api
                # Probably want to add support for using quantized tensors more directly
                xk = self.cache_quantizer.quantize(xk).unpack().qs
                xv = self.cache_quantizer.quantize(xv).unpack().qs

        xk, xv = self.transact_cache(
            xk_cache_update=xk,
            xv_cache_update=xv,
            seq_block_ids=seq_block_ids,
            kv_seq_len=kv_seq_len,
            start_positions=start_positions,
            cache_state=cache_state,
        )

        # Expand kv heads for GQA.
        gqa_n_rep = self.head_count // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:

            def repeat_kv(x: torch.Tensor) -> torch.Tensor:
                bs, slen, n_kv_heads, head_dim = x.shape
                unsq = x.unsqueeze(-2)
                exp = ops.expand(unsq, (bs, slen, n_kv_heads, gqa_n_rep, head_dim))
                return exp.flatten(2, 3)

            xk = repeat_kv(xk)
            xv = repeat_kv(xv)

        # Fake quant is already dequantized when stored in the cache.
        if self.cache_quantizer and not self.fake_quant:
            xk = self.cache_quantizer.dequantize_raw_tensor(
                xk, self.attention_dtype, name="xk_deq"
            )
            xv = self.cache_quantizer.dequantize_raw_tensor(
                xv, self.attention_dtype, name="xv_deq"
            )
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.attention_dtype)

        # Transpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        if self.attention_kernel == "decomposed":
            attn_weights = ops.matmul(xq, keys.transpose(2, 3))
            if self.attention_scale is None:
                attn_weights = attn_weights / math.sqrt(self.head_dim)
            else:
                attn_weights = attn_weights * self.attention_scale

            # Flash attention.
            if self.softcap is not None:
                attn_weights = self.softcap * torch.tanh(attn_weights / self.softcap)

            self.assert_not_nan(attn_weights)

            # Apply attention mask.
            self.trace_tensor("attn_weights", attn_weights)
            if attention_mask is not None:
                # self.trace_tensor("attn_mask", attention_mask)
                attn_weights = attn_weights + attention_mask

            attn_weights = ops.softmax(
                ops.to(attn_weights, dtype=torch.float32), dim=-1
            )
            attn_weights = ops.to(attn_weights, dtype=xq.dtype)
            attn_output = ops.matmul(
                attn_weights, values
            )  # (bs, heads, slen, head_dim)
        else:
            attn_output = ops.scaled_dot_product_attention(
                q=xq,  # [bs, ..., sl, dim]
                k=keys,  # [bs, ..., sl, dim]
                v=values,  # [bs, ..., sl, dim]
                a=attention_mask,  # [bs, ..., sl, sl]
                is_causal=attention_mask is None,  # assumes causal masking when true
                scale=None,  # defaults to 1/sqrt(dim)
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(2, 3)
        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output
        return h

    def transact_cache(
        self,
        *,
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        cache_state: list[torch.Tensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: Optional[torch.Tensor],
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
    ):
        cache = self.cache
        # Manage the cache.
        if start_positions is None:
            # Prefill: Write the entire cache.
            cache.write(
                cache_state,
                cache_partitions=[xk_cache_update, xv_cache_update],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )
            return xk_cache_update, xv_cache_update

        # Decode at ragged start positions.
        # We need to initialize/read the K/V from the cache for the whole
        # sequence. Note that at this point, it is possible to fork and
        # use a memory efficient attention kernel that can do indirect
        # reads, skipping this materialization. This path is taken for
        # a decode step.
        assert xk_cache_update.shape[1] == 1
        assert xv_cache_update.shape[1] == 1
        assert kv_seq_len == seq_block_ids.shape[1] * cache.block_seq_stride

        # Write our one updated cache row into the cache.
        cache.write_timestep(
            cache_state,
            cache_partitions=[
                xk_cache_update,
                xv_cache_update,
            ],
            transformer_block_index=self.block_index,
            seq_positions=start_positions,
            page_ids=seq_block_ids,
        )

        # Restore from the cache.
        xk, xv = cache.read(
            cache_state,
            transformer_block_index=self.block_index,
            page_ids=seq_block_ids,
            seq_len=kv_seq_len,
        )

        # For computation, we create a subview of the xk/xv tensors to have
        # a sequence length covering the blocked size. This must include
        # the newly added row (the caller is responsible for ensuring that
        # every block has at least one row left). We'll compute on this
        # ragged view and use an appropriate mask.
        return xk, xv
