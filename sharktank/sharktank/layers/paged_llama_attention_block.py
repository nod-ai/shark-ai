# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types import *
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .rotary_embedding import RotaryEmbeddingLayer
from .paged_attention import PagedAttention
from sharktank import ops

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
        model_arch: str,
        attention_kernel: str = "torch",
        v_head_dim: Optional[int] = None,
        rope_dimension_count: Optional[int] = None,
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
        block_to_pipeline_map: tuple[tuple[int, ...], ...] | None = None,
        pipeline_to_device_map: tuple[tuple[int, ...], ...] | None = None,
    ):
        super().__init__(theta)

        self.paged_attention = PagedAttention(
            transformer_block_count=cache.transformer_block_count,
            block_to_pipeline_map=block_to_pipeline_map,
            pipeline_to_device_map=pipeline_to_device_map,
            attn_head_count=head_count_kv,
            attn_head_dim=head_dim,
            block_seq_stride=cache.block_seq_stride,
            cache_dtype=cache.cache_dtype,
            attn_dtype=cache.attn_dtype,
            device=cache.device,
            shard_count=cache.shard_count,
        )
        self.shard_count = cache.shard_count
        self.paged_attention = cache
        self.block_index = block_index
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_kernel = attention_kernel
        self.attention_scale = attention_scale
        self.rope_dimension_count = rope_dimension_count
        self.softcap = softcap
        self.fake_quant = fake_quant
        self.cache_quantizer = None
        self.probs_quantizer = None
        self.model_arch = model_arch
        self.v_head_dim = v_head_dim

        self.attn_type_map = {
            "llama": "gqa",
            "grok": "gqa",
            "deepseek2": "mla",
        }
        self.attn_type = self.attn_type_map[self.model_arch]

        if self.attn_type == "gqa":
            self.add_module(
                "attn_q", LinearLayer(theta("attn_q"), fake_quant=self.fake_quant)
            )
            self.add_module(
                "attn_k", LinearLayer(theta("attn_k"), fake_quant=self.fake_quant)
            )
            self.add_module(
                "attn_v", LinearLayer(theta("attn_v"), fake_quant=self.fake_quant)
            )
        elif self.attn_type == "mla":
            self.add_module(
                "kv_norm", RMSNormLayer(theta("attn_kv_a_norm"), epsilon=rms_epsilon)
            )
            if "q" in theta:
                self.wq = LinearLayer(theta("q"), fake_quant=self.fake_quant)
            else:
                self.wq = None
                self.wq_a = LinearLayer(theta("attn_q_a"), fake_quant=self.fake_quant)
                self.wq_b = LinearLayer(theta("attn_q_b"), fake_quant=self.fake_quant)
                self.q_norm = RMSNormLayer(theta("attn_q_a_norm"), epsilon=rms_epsilon)

            self.add_module(
                "wkv_a", LinearLayer(theta("attn_kv_a_mqa"), fake_quant=self.fake_quant)
            )
            self.add_module(
                "wkv_b", LinearLayer(theta("attn_kv_b"), fake_quant=self.fake_quant)
            )

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
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

    def pre_process_attention(
        self,
        x: torch.Tensor,
        start_index: int,
        embedding: RotaryEmbeddingLayer,
        embedding_batch_mask: torch.Tensor,
    ):

        if self.attn_type == "mla":
            if self.wq is not None:
                q = self.wq(x).unflatten(2, (self.head_count, -1))
            else:
                q = self.wq_b(self.q_norm(self.wq_a(x)))
                if isinstance(q, ShardedTensor) and not isinstance(q, ReplicatedTensor):
                    q = ops.replicate(q, count=self.shard_count)
                q = q.unflatten(2, (self.head_count, -1))

            qk_nope_head_dim = q.shape[-1] - self.rope_dimension_count
            q_nope = q[:, :, :, :qk_nope_head_dim]
            q_rope = q[:, :, :, qk_nope_head_dim:]

            kv = self.wkv_a(x)
            kv_nope_size = kv.shape[-1] - self.rope_dimension_count
            kv_nope = kv[:, :, :kv_nope_size]
            k_rope = kv[:, :, kv_nope_size:]

            if start_index is not None:
                q_rope = embedding(xt=q_rope, start_index=start_index)
                k_rope = embedding(xt=k_rope.unsqueeze(2), start_index=start_index)
            else:
                q_rope = embedding.apply_batched_mask(
                    xt=q_rope, mask=embedding_batch_mask
                )
                k_rope = embedding.apply_batched_mask(
                    xt=k_rope.unsqueeze(2), mask=embedding_batch_mask
                )

            xq = ops.cat((q_nope, q_rope), dim=-1)

            ##TODO: Restructure this to apply the wkv_b post attention instead of here
            kv_norm = self.kv_norm(kv_nope)
            wkv_b = self.wkv_b(kv_norm)
            if isinstance(wkv_b, ShardedTensor) and not isinstance(
                wkv_b, ReplicatedTensor
            ):
                wkv_b = ops.replicate(wkv_b, count=self.shard_count)
            wkv_b = wkv_b.unflatten(2, (self.head_count, -1))

            k_nope = wkv_b[:, :, :, :qk_nope_head_dim]
            xv = wkv_b[:, :, :, qk_nope_head_dim:]

            k_rope = ops.repeat(k_rope, (1, 1, k_nope.shape[2] // k_rope.shape[2], 1))

            if isinstance(k_rope, ReplicatedTensor) and isinstance(
                k_nope, SplitPrimitiveTensor
            ):
                k_rope = ops.reshard_split(
                    k_rope, dim=k_nope.shard_dim, count=k_nope.shard_count
                )

            xk = ops.cat((k_nope, k_rope), dim=-1)

        elif self.attn_type == "gqa":
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

        return xq, xk, xv

    def forward(
        self,
        h: torch.Tensor | ShardedTensor,
        *,
        embedding: RotaryEmbeddingLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor | ReplicatedTensor] = None,
        attention_mask: Optional[torch.Tensor | ReplicatedTensor] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
    ):
        assert bool(start_index is not None) ^ bool(embedding_batch_mask is not None)

        x = self.attn_norm(h)

        xq, xk, xv = self.pre_process_attention(
            x, start_index, embedding, embedding_batch_mask
        )

        # Used by fp8_e4m3fnuz model
        if self.cache_quantizer is not None:
            if not self.fake_quant:
                # TODO: this seems like a bastardization of our quantized tensor api
                # Probably want to add support for using quantized tensors more directly
                xk = self.cache_quantizer.quantize(xk).unpack().qs
                xv = self.cache_quantizer.quantize(xv).unpack().qs

        # Pad final dim of v to match with kv cache
        if self.attn_type == "mla" and self.head_dim != self.v_head_dim:
            xv = ops.pad(xv, [0, self.head_dim - self.v_head_dim])

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
                start_positions=start_positions,
                head_count_attn=self.head_count,
                cache_quantizer=self.cache_quantizer,
                fake_quant=self.fake_quant,
                attention_kernel=self.attention_kernel,
                mask=attention_mask,
                scale=self.attention_scale,
                softcap=self.softcap,
            )
        # attn_output is sharded
        # Drop padded part of attn_output
        if self.attn_type == "mla" and self.head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.transpose(1, 2)

        if self.attn_type == "mla":
            attn_output = attn_output.flatten(2)
        else:
            attn_output = attn_output.flatten(2, 3)

        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output
        return h
