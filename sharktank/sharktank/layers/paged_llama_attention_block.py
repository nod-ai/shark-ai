# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.layers import CachedRotaryLayer
from sharktank.layers.configs.llm_configs import LlamaModelConfig
from sharktank.types import *
from sharktank.utils.create_cache import create_paged_attention
from sharktank.utils.attention import *
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer, L2Norm
from .latent_attention_block import LatentAttentionBlock
from .paged_attention import CacheAllocation, attn_type_map
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
        config: LlamaModelConfig,
        block_index: int,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        model_arch: str,
        matmul_kernel: Optional[str] = None,
        v_head_dim: Optional[int] = None,
        rope_dimension_count: Optional[int] = None,
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
        attn_temperature_tuning: bool = False,
        floor_scale: Optional[float] = None,
    ):
        super().__init__(theta)

        attention_kernel = (
            "decomposed" if config.hp.model_arch == "grok" else config.attention_kernel
        )

        if config.hp.model_arch == "llama4":
            use_rope = (
                block_index in config.rope_layers if config.rope_layers else False
            )
            use_qk_norm = use_rope and config.use_qk_norm
        else:
            use_rope = True
            use_qk_norm = False

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
        self.model_arch = model_arch
        self.v_head_dim = v_head_dim
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_type = attn_type_map[self.model_arch]

        if self.attn_type == "gqa":
            self.add_module(
                "attn_q",
                LinearLayer(
                    theta("attn_q"),
                    fake_quant=self.fake_quant,
                    matmul_kernel=matmul_kernel,
                ),
            )
            self.add_module(
                "attn_k",
                LinearLayer(
                    theta("attn_k"),
                    fake_quant=self.fake_quant,
                    matmul_kernel=matmul_kernel,
                ),
            )
            self.add_module(
                "attn_v",
                LinearLayer(
                    theta("attn_v"),
                    fake_quant=self.fake_quant,
                    matmul_kernel=matmul_kernel,
                ),
            )
            self.paged_attention = create_paged_attention(
                config, self.attn_k.q_output, self.attn_v.q_output
            )
        elif self.attn_type == "mla":
            self.add_module(
                "latent_attn",
                LatentAttentionBlock(
                    theta,
                    rms_epsilon=rms_epsilon,
                    head_count=self.head_count,
                    head_count_kv=self.head_count_kv,
                    rope_dimension_count=self.rope_dimension_count,
                    fake_quant=self.fake_quant,
                ),
            )
            self.paged_attention = create_paged_attention(config)

        self.paged_attention.attention_chunk_size = config.attention_chunk_size
        self.paged_attention.use_attention_mask = config.use_attention_mask
        self.paged_attention.use_

        if self.use_qk_norm:
            self.qk_norm = L2Norm(dim=-1, epsilon=rms_epsilon)

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module(
            "attn_output",
            LinearLayer(
                theta("attn_output"),
                fake_quant=self.fake_quant,
                matmul_kernel=matmul_kernel,
            ),
        )
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

    def gqa_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[InferenceTensor],
    ):
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

        if self.use_rope:
            xq = embedding.apply_batched_mask(xt=xq, start_positions=start_positions)
            xk = embedding.apply_batched_mask(xt=xk, start_positions=start_positions)

        if self.attn_q.q_output is not None:
            xq = ops.quantize(xq, self.attn_q.q_output)
        if self.attn_k.q_output is not None:
            xk = ops.quantize(xk, self.attn_k.q_output)
        if self.attn_v.q_output is not None:
            xv = ops.quantize(xv, self.attn_v.q_output)
        return xq, xk, xv

    def pre_process_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[torch.Tensor],
        is_decode: bool,
    ):
        """
        x:
            input token embeddings.
            shape is (batch_size, sequence_length, embedding_length)
        """
        if self.attn_type == "gqa":
            xq, xk, xv = self.gqa_attention(
                x,
                embedding=embedding,
                start_positions=start_positions,
            )

        elif self.attn_type == "mla":
            xq, xk, xv = self.latent_attn(
                x,
                embedding=embedding,
                is_decode=is_decode,
            )

        return xq, xk, xv

    def forward(
        self,
        h: torch.Tensor | ShardedTensor,
        *,
        embedding: CachedRotaryLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        start_positions: Optional[torch.Tensor] = None,
        cache_state: CacheAllocation | None = None,
    ):
        is_decode = isinstance(h.shape[1], int) and h.shape[1] == 1
        if is_decode:
            # Precompute a position based mask for computing rope embeddings
            # as it is the same for all blocks.
            # TODO: What is start_positions here if running with MLA?
            embedding_batch_mask = embedding.compute_batch_mask(
                start_positions, batch_seq_len=1
            )
            self.trace_tensor("llama.embedding_batch_mask", embedding_batch_mask)

            input_mask = create_input_mask(
                seq_lens,
                seq_block_ids.shape[1] * self.paged_attention.block_seq_stride,
            )
            attention_mask = create_attention_mask_for_decode(
                input_mask, embedding._dtype
            )
        else:
            embedding_batch_mask = None
            attention_mask = None
            if self.paged_attention.use_attention_mask:
                input_mask = create_input_mask(seq_lens, h.shape[1])
                attention_mask = create_attention_mask(
                    input_mask,
                    self.model.activation_dtype,
                    start_positions=start_positions,
                )
            use_chunked_attention_mask = (
                self.paged_attention.attention_chunk_size is not None
            )
            if use_chunked_attention_mask and self.use_rope:
                attention_mask = create_chunked_attention_mask(
                    attention_mask, self.paged_attention.attention_chunk_size
                )
            # Need attention_chunk_size
            # Need block_idx
            # Need rope_layers
            # <=> use rope

        x = self.attn_norm(h)

        xq, xk, xv = self.pre_process_attention(
            x, embedding, start_positions, embedding_batch_mask
        )

        if self.use_qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399
        # Ken M. Nakanishi - Scalable-Softmax Is Superior for Attention (2025)
        if self.attn_temperature_tuning and not self.use_rope:
            if start_positions is None:
                cache_position = torch.arange(
                    0, h.shape[1], dtype=torch.long, device=h.device
                )
            else:
                assert False, "TODO: decode step"
            attn_scales = (
                torch.log(
                    torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0
                )
                * self.attention_scale
                + 1.0
            ).to(xq.device)
            input_tokens_shape = h.shape[:-1]
            attn_scales = attn_scales.view((1, input_tokens_shape[-1], 1, 1)).expand(
                (*input_tokens_shape, 1, 1)
            )  # batch size > 1
            xq = (xq * attn_scales).to(xq.dtype)

        # Used by fp8_e4m3fnuz model
        if self.cache_quantizer is not None:
            if not self.fake_quant:
                # TODO: this seems like a bastardization of our quantized tensor api
                # Probably want to add support for using quantized tensors more directly
                xk = ops.unpack(ops.quantize(xk, self.cache_quantizer)).qs
                xv = ops.unpack(ops.quantize(xv, self.cache_quantizer)).qs

        # Pad final dim of v to match with kv cache
        if self.attn_type == "mla" and self.head_dim != self.v_head_dim:
            xv = ops.pad(xv, [0, self.head_dim - self.v_head_dim])

        if not is_decode:
            attn_output = self.paged_attention.forward_prefill(
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

        h = h + attn_output.to(dtype=h.dtype)
        return h
