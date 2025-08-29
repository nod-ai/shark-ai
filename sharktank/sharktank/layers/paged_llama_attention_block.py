# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from abc import ABC, abstractmethod
from sharktank.layers import CachedRotaryLayer
from sharktank.types import *
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer, L2Norm
from .latent_attention_block import LatentAttentionBlock
from .paged_attention import CacheAllocation, PagedAttention, attn_type_map
from sharktank import ops

__all__ = [
    "PagedLlamaAttentionBlockGqa",
    "PagedLlamaAttentionBlockMla",
]


class PagedLlamaAttentionBlockBase(ThetaLayer):
    """
    Abstract base class for Llama-style attention blocks using paged KV cache.

    This class encapsulates shared configuration, quantization, and attention application logic
    for different attention mechanisms (e.g., GQA and MLA). Subclasses must implement
    `pre_process_attention` and `forward`.

    Attributes:
        paged_attention (PagedAttention): The paged attention mechanism.
        block_index (int): Index of the current block in the model.
        head_count (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        head_count_kv (int): Number of key/value heads.
        model_arch (str): Model architecture identifier.
        attn_type (str): Attention type derived from model architecture.
        Various optional parameters for quantization, scaling, and rotary embeddings.
    """
    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        paged_attention: PagedAttention,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        model_arch: str,
        attention_kernel: Optional[str] = "torch",
        matmul_kernel: Optional[str] = None,
        v_head_dim: Optional[int] = None,
        rope_dimension_count: Optional[int] = None,
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        attn_temperature_tuning: bool = False,
        floor_scale: Optional[float] = None,
    ):
        super().__init__(theta)
        self.paged_attention = paged_attention
        self.block_index = block_index
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_kernel = attention_kernel
        self.matmul_kernel = matmul_kernel
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
        assert (
            self.attn_type == self.paged_attention.attn_type
        ), f"Attention type mismatch: {self.attn_type} != {self.paged_attention.attn_type}"

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def pre_process_attention(...):
        ...

    def _apply_attention(self, xq, xk, xv, h, seq_block_ids, start_positions, attention_mask, cache_state):
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

        is_decode = isinstance(h.shape[1], int) and h.shape[1] == 1
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
                k_quantizer=self.k_quantizer,
                v_quantizer=self.v_quantizer,
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
                k_quantizer=self.k_quantizer,
                v_quantizer=self.v_quantizer,
            )
        return attn_output

class PagedLlamaAttentionBlockGqa(PagedLlamaAttentionBlockBase):
    """
    Implements Grouped Query Attention (GQA) variant of the Llama attention block.

    This class initializes and applies linear projections for query, key, and value tensors,
    and handles rotary embeddings and quantization specific to GQA.

    Raises:
        ValueError: If `attn_type` is not 'gqa'.

    Methods:
        pre_process_attention: Projects input to Q, K, V and applies rotary embeddings.
        forward: Applies attention and outputs the transformed tensor.
    """
    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        paged_attention: PagedAttention,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        model_arch: str,
        attention_kernel: Optional[str] = "torch",
        matmul_kernel: Optional[str] = None,
        v_head_dim: Optional[int] = None,
        rope_dimension_count: Optional[int] = None,
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        attn_temperature_tuning: bool = False,
        floor_scale: Optional[float] = None,
    ):
        super().__init__(
        theta,
        block_index=block_index,
        paged_attention=paged_attention,
        head_count=head_count,
        head_dim=head_dim,
        head_count_kv=head_count_kv,
        rms_epsilon=rms_epsilon,
        model_arch=model_arch,
        attention_kernel=attention_kernel,
        matmul_kernel=matmul_kernel,
        v_head_dim=v_head_dim,
        rope_dimension_count=rope_dimension_count,
        attention_scale=attention_scale,
        softcap=softcap,
        fake_quant=fake_quant,
        use_rope=use_rope,
        use_qk_norm=use_qk_norm,
        attn_temperature_tuning=attn_temperature_tuning,
        floor_scale=floor_scale,)

        if self.attn_type != "gqa":
            raise ValueError(f"{self.__class__.__name__} requires attn_type='gqa', got '{self.attn_type}'")

        for name in ["attn_q", "attn_k", "attn_v"]:
            self.add_module(
                name,
                LinearLayer(theta(name), fake_quant=self.fake_quant, matmul_kernel=matmul_kernel),
            )

        self.k_quantizer = self.attn_k.q_output
        self.v_quantizer = self.attn_v.q_output

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

    def _gqa_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[InferenceTensor],
        embedding_batch_mask: tuple[InferenceTensor, InferenceTensor] | None,
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
            # Fast path to start_index based embedding lookup if available.
            # Falls back to a slower position based index lookup.
            if embedding_batch_mask is None:
                xq = embedding.forward(xt=xq, start_positions=start_positions)
                xk = embedding.forward(xt=xk, start_positions=start_positions)
            else:
                xq = embedding.apply_batched_mask(xt=xq, mask=embedding_batch_mask)
                xk = embedding.apply_batched_mask(xt=xk, mask=embedding_batch_mask)

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
        embedding_batch_mask: tuple[InferenceTensor, InferenceTensor] | None,
    ):
        xq, xk, xv = self._gqa_attention(
            x,
            embedding=embedding,
            start_positions=start_positions,
            embedding_batch_mask=embedding_batch_mask,
        )

        return xq, xk, xv

    def forward(
        self,
        h: torch.Tensor | ShardedTensor,
        *,
        embedding: CachedRotaryLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: None | tuple[InferenceTensor, InferenceTensor] = None,
        cache_state: CacheAllocation | None = None,
    ):
        x = self.attn_norm(h)

        xq, xk, xv = self.pre_process_attention(
            x, embedding, start_positions, embedding_batch_mask
        )

        if self.use_qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)

        attn_output = _apply_attention(xq, xk, xv, h, seq_block_ids, start_positions, attention_mask, cache_state)

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.flatten(2, 3)

        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output.to(dtype=h.dtype)
        return h

class PagedLlamaAttentionBlockMla(PagedLlamaAttentionBlockBase):
    """
    Implements Multi-Latent Attention (MLA) variant of the Llama attention block.

    This class uses a latent attention block to generate Q, K, V representations and
    handles reshaping and padding specific to MLA.

    Raises:
        ValueError: If `attn_type` is not 'mla'.

    Methods:
        pre_process_attention: Uses latent attention block to compute Q, K, V.
        forward: Applies attention and outputs the transformed tensor.
    """
    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        paged_attention: PagedAttention,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        rms_epsilon: float,
        model_arch: str,
        attention_kernel: Optional[str] = "torch",
        matmul_kernel: Optional[str] = None,
        v_head_dim: Optional[int] = None,
        rope_dimension_count: Optional[int] = None,
        attention_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fake_quant: Optional[bool] = True,
        use_rope: bool = True,
        use_qk_norm: bool = False,
        attn_temperature_tuning: bool = False,
        floor_scale: Optional[float] = None,
    ):
        super().__init__(
        theta,
        block_index=block_index,
        paged_attention=paged_attention,
        head_count=head_count,
        head_dim=head_dim,
        head_count_kv=head_count_kv,
        rms_epsilon=rms_epsilon,
        model_arch=model_arch,
        attention_kernel=attention_kernel,
        matmul_kernel=matmul_kernel,
        v_head_dim=v_head_dim,
        rope_dimension_count=rope_dimension_count,
        attention_scale=attention_scale,
        softcap=softcap,
        fake_quant=fake_quant,
        use_rope=use_rope,
        use_qk_norm=use_qk_norm,
        attn_temperature_tuning=attn_temperature_tuning,
        floor_scale=floor_scale,)
        if self.attn_type != "mla":
            raise ValueError(f"{self.__class__.__name__} requires attn_type='mla', got '{self.attn_type}'")

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

    def pre_process_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[torch.Tensor],
        embedding_batch_mask: tuple[InferenceTensor, InferenceTensor] | None,
    ):
        xq, xk, xv = self.latent_attn(
            x,
            embedding=embedding,
            embedding_batch_mask=embedding_batch_mask,
        )

        return xq, xk, xv

    def forward(
        self,
        h: torch.Tensor | ShardedTensor,
        *,
        embedding: CachedRotaryLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: None | tuple[InferenceTensor, InferenceTensor] = None,
        cache_state: CacheAllocation | None = None,
    ):
        x = self.attn_norm(h)

        xq, xk, xv = self.pre_process_attention(
            x, embedding, start_positions, embedding_batch_mask
        )

        if self.use_qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)

        attn_output = self._apply_attention(xq, xk, xv, h, seq_block_ids, start_positions, attention_mask, cache_state)

        # attn_output is sharded
        # Drop padded part of attn_output
        if self.head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.flatten(2)

        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output.to(dtype=h.dtype)
        return h
