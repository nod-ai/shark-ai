# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional


import torch
from ..types import QuantizerTensor, StaticScaledQuantizer
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .rotary_embedding import RotaryEmbeddingLayer
from .paged_attention import PagedAttention

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
        cache: PagedAttention,
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

        self.paged_attention = PagedAttention(
            transformer_block_count=cache.transformer_block_count,
            attn_head_count=head_count_kv,
            attn_head_dim=head_dim,
            block_seq_stride=cache.block_seq_stride,
            dtype=cache.dtype,
            device=cache.device,
            shard_count=cache.shard_count,
        )
        self.block_index = block_index
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_dtype = attention_dtype
        self.attention_kernel = attention_kernel
        self.attention_scale = attention_scale
        self.softcap = softcap
        self.fake_quant = fake_quant
        self.cache_quantizer = None
        self.probs_quantizer = None

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
        if "kv_cache" in theta.keys:
            self.cache_quantizer: Optional[QuantizerTensor] = theta.optional_tensor(
                "kv_cache.quantizer"
            )
        if "attn_scale" in theta.keys:
            self.attention_scale = theta("attn_scale").as_torch()
            self.probs_quantizer = StaticScaledQuantizer(
                name="attn_scale.quantizer",
                scale=1.0 / (self.attention_scale * 2.0),
                reciprocal_scale=self.attention_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
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
        kv_seq_len = seq_block_ids.shape[1] * self.paged_attention.block_seq_stride

        # Used by fp8_e4m3fnuz model
        if self.cache_quantizer is not None:
            if not self.fake_quant:
                # TODO: this seems like a bastardization of our quantized tensor api
                # Probably want to add support for using quantized tensors more directly
                xk = self.cache_quantizer.quantize(xk).unpack().qs
                xv = self.cache_quantizer.quantize(xv).unpack().qs

        if start_positions is None:
            attn_output = self.paged_attention.forward_prefill(
                q=xq,
                k=xk,
                v=xv,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
                block_index=self.block_index,
                head_count_attn=self.head_count,
                cache_quantizer=self.cache_quantizer,
                fake_quant=self.fake_quant,
                attention_kernel=self.attention_kernel,
                mask=attention_mask,
                scale=self.attention_scale,
                softcap=self.softcap,
                probs_quantizer=self.probs_quantizer,
            )
        else:
            attn_output = self.paged_attention.forward_decode(
                q=xq,
                k=xk,
                v=xv,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
                block_index=self.block_index,
                kv_seq_len=kv_seq_len,
                start_positions=start_positions,
                head_count_attn=self.head_count,
                cache_quantizer=self.cache_quantizer,
                fake_quant=self.fake_quant,
                attention_kernel=self.attention_kernel,
                mask=attention_mask,
                scale=self.attention_scale,
                softcap=self.softcap,
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
        seq_block_ids: torch.Tensor,
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
    ):
        # Manage the cache.
        if start_positions is None:
            # Prefill: Write the entire cache.
            self.paged_attention.write(
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
        assert (
            kv_seq_len == seq_block_ids.shape[1] * self.paged_attention.block_seq_stride
        )

        # Write our one updated cache row into the cache.
        self.paged_attention.write_timestep(
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
        xk, xv = self.paged_attention.read(
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
