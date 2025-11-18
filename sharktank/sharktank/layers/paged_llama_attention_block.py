# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Optional
import logging

import torch

from amdsharktank.layers import CachedRotaryLayer
from amdsharktank.layers.configs.llm_configs import LlamaModelConfig
from amdsharktank.types import *
from amdsharktank.utils.create_cache import create_paged_attention
from amdsharktank.utils.attention import *
from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer, L2Norm
from .latent_attention_block import LatentAttentionBlock
from .kv_cache import CacheAllocation, KVCache
from .paged_attention import attn_type_map
from amdsharktank import ops


logger = logging.getLogger(__name__)


class PagedLlamaAttentionBlock(ABC, ThetaLayer):
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
        kv_cache: KVCache,
        attention_kernel: Optional[str],
        matmul_kernel: Optional[str],
        v_head_dim: Optional[int],
        rope_dimension_count: Optional[int],
        attention_scale: Optional[float],
        softcap: Optional[float],
        fake_quant: Optional[bool],
        use_rope: bool,
        use_qk_norm: bool,
        attn_temperature_tuning: bool,
        floor_scale: Optional[float],
        dims_to_flatten: tuple[int, ...],
        sliding_window: Optional[int] = None,
        use_fused_qkv: bool = False,
    ):
        super().__init__(theta)

        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_kernel = attention_kernel
        self.attention_scale = attention_scale
        self.rope_dimension_count = rope_dimension_count
        self.softcap = softcap
        self.fake_quant = fake_quant
        self.matmul_kernel = matmul_kernel
        self.v_head_dim = v_head_dim
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.attn_temperature_tuning = attn_temperature_tuning
        self.kv_cache = kv_cache
        self.floor_scale = floor_scale
        self.dims_to_flatten = dims_to_flatten
        self.rms_epsilon = rms_epsilon
        self.sliding_window = sliding_window
        self.use_fused_qkv = use_fused_qkv

        self.cache_quantizer = None
        if "kv_cache" in theta.keys:
            self.cache_quantizer: Optional[QuantizerTensor] = theta.optional_tensor(
                "kv_cache.quantizer"
            )

        self.q_quantizer: QuantizerTensor | None = None
        self.k_quantizer: QuantizerTensor | None = None
        self.v_quantizer: QuantizerTensor | None = None

        if self.use_fused_qkv:
            self.add_module(
                "attn_qkv", LinearLayer(theta("attn.wqkv"), fake_quant=self.fake_quant)
            )
            self.k_quantizer = None
            self.v_quantizer = None
        else:
            for attn_var in ["q", "k", "v"]:
                attn_name = f"attn_{attn_var}"

                if attn_name not in theta:
                    print(f"  {attn_name} NOT found in theta, skipping")
                    continue

                self.add_module(
                    attn_name,
                    LinearLayer(
                        theta(attn_name),
                        fake_quant=self.fake_quant,
                        matmul_kernel=self.matmul_kernel,
                    ),
                )
                setattr(
                    self,
                    f"{attn_var}_quantizer",
                    theta.optional_tensor(f"{attn_name}.q_output"),
                )

        if "attn_sinks" in theta.keys:
            sink_tensor = theta("attn_sinks")
            ops.module_register_buffer(self, "sink", sink_tensor)
        else:
            self.sink = None

        self.paged_attention = create_paged_attention(
            config,
            kv_cache,
            use_rope,
            block_index,
            self.k_quantizer,
            self.v_quantizer,
        )

        if self.use_qk_norm:
            self.qk_norm = L2Norm(dim=-1, epsilon=self.rms_epsilon)

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=self.rms_epsilon)
        )
        self.add_module(
            "attn_output",
            LinearLayer(
                theta("attn_output"),
                fake_quant=self.fake_quant,
                matmul_kernel=self.matmul_kernel,
            ),
        )

        if theta.optional_tensor("attn_output_norm") is None:
            self.add_module(
                "attn_output_norm",
                torch.nn.Identity(),
            )
        else:
            self.add_module(
                "attn_output_norm",
                RMSNormLayer(theta("attn_output_norm"), epsilon=self.rms_epsilon),
            )

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
        x = self.attn_norm(h)

        xq, xk, xv = self.pre_process_attention(x, embedding, start_positions)

        if self.use_qk_norm:
            xq = self.qk_norm(xq)
            xk = self.qk_norm(xk)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399
        # Ken M. Nakanishi - Scalable-Softmax Is Superior for Attention (2025)
        if self.attn_temperature_tuning and not self.use_rope:
            if start_positions is None:
                cache_position = ops.arange(
                    0, h.shape[1], dtype=torch.long, device=h.device
                )
            else:
                assert False, "TODO: decode step"
            attn_scales = (
                ops.log(
                    torch.floor(
                        (cache_position.to(torch.float32) + 1.0) / self.floor_scale
                    )
                    + 1.0
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
        if self.cache_quantizer and not self.fake_quant:
            # TODO: this seems like a bastardization of our quantized tensor api
            # Probably want to add support for using quantized tensors more directly
            xk = ops.unpack_to_qs(ops.quantize(xk, self.cache_quantizer))
            xv = ops.unpack_to_qs(ops.quantize(xv, self.cache_quantizer))

        xv = self.pad_kv(xv)

        is_decode = isinstance(h.shape[1], int) and h.shape[1] == 1
        if is_decode:
            attn_function = self.paged_attention.forward_decode
        else:
            attn_function = self.paged_attention.forward_prefill
        attn_output = attn_function(
            q=xq,
            k=xk,
            v=xv,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            start_positions=start_positions,
            head_count_attn=self.head_count,
            cache_quantizer=self.cache_quantizer,
            fake_quant=self.fake_quant,
            attention_kernel=self.attention_kernel,
            scale=self.attention_scale,
            softcap=self.softcap,
            seq_lens=seq_lens,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )
        attn_output = self.unpad_attn_output(attn_output)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(*self.dims_to_flatten)

        # Project.
        attn_output = self.attn_output(attn_output)
        attn_output = self.attn_output_norm(attn_output)

        h = h + attn_output.to(dtype=h.dtype)
        return h

    @abstractmethod
    def pre_process_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor | ReplicatedTensor,
        torch.Tensor | ReplicatedTensor,
        torch.Tensor | ReplicatedTensor,
    ]:
        ...

    @abstractmethod
    def pad_kv(
        self, xv: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        """Pad the KV cache to match the head dimension."""
        ...

    @abstractmethod
    def unpad_attn_output(
        self, attn_output: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        """Unpad the attention output to match the head dimension."""
        ...


class PagedLlamaGQAttentionBlock(PagedLlamaAttentionBlock):
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
        kv_cache: KVCache,
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
        sliding_window: Optional[int] = None,
        use_fused_qkv: bool = False,
    ):
        super().__init__(
            theta=theta,
            config=config,
            block_index=block_index,
            head_count=head_count,
            head_dim=head_dim,
            head_count_kv=head_count_kv,
            rms_epsilon=rms_epsilon,
            kv_cache=kv_cache,
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
            floor_scale=floor_scale,
            dims_to_flatten=(2, 3),
            sliding_window=sliding_window,
            use_fused_qkv=use_fused_qkv,
        )

    def pad_kv(
        self, xv: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        # Note needed for GQA
        return xv

    def unpad_attn_output(
        self, attn_output: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        # Note needed for GQA
        return attn_output

    def _project_qkv(self, x):
        bs, batch_seq_len, _ = x.shape
        if self.use_fused_qkv:
            # Fused QKV path: single linear layer + slicing
            qkv = self.attn_qkv(x)

            # Slice QKV into separate tensors
            q_end = self.head_count * self.head_dim
            k_end = q_end + self.head_count_kv * self.head_dim
            v_end = k_end + self.head_count_kv * self.head_dim

            q = qkv[:, :, :q_end]
            k = qkv[:, :, q_end:k_end]
            v = qkv[:, :, k_end:v_end]
        else:
            q = self.attn_q(x)
            k = self.attn_k(x)
            v = self.attn_v(x)
        assert q.shape[-1] == self.head_count * self.head_dim
        assert k.shape[-1] == self.head_count_kv * self.head_dim
        assert v.shape[-1] == self.head_count_kv * self.head_dim

        xq = q.view(bs, batch_seq_len, self.head_count, self.head_dim)
        xk = k.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)
        xv = v.view(bs, batch_seq_len, self.head_count_kv, self.head_dim)
        return xq, xk, xv

    def pre_process_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[torch.Tensor],
    ):

        xq, xk, xv = self._project_qkv(x)

        if self.use_rope:
            xq = embedding.forward(xt=xq, start_positions=start_positions)
            xk = embedding.forward(xt=xk, start_positions=start_positions)

        if (
            not self.use_fused_qkv
        ):  # TODO: we need to add quantization for the fused qkv path
            # For separate QKV, apply individual quantization
            if self.attn_q.q_output is not None:
                xq = ops.quantize(xq, self.attn_q.q_output)
            if self.attn_k.q_output is not None:
                xk = ops.quantize(xk, self.attn_k.q_output)
            if self.attn_v.q_output is not None:
                xv = ops.quantize(xv, self.attn_v.q_output)
        return xq, xk, xv


class PagedLlamaMLAttentionBlock(PagedLlamaAttentionBlock):
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
        kv_cache: KVCache,
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
        floor_scale: Optional[float],
        sliding_window: Optional[int] = None,
        use_fused_qkv: bool = False,
    ):
        super().__init__(
            theta=theta,
            config=config,
            block_index=block_index,
            head_count=head_count,
            head_dim=head_dim,
            head_count_kv=head_count_kv,
            rms_epsilon=rms_epsilon,
            kv_cache=kv_cache,
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
            floor_scale=floor_scale,
            dims_to_flatten=(2,),
            sliding_window=sliding_window,
            use_fused_qkv=use_fused_qkv,
        )

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

    def pad_kv(
        self, xv: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        if self.head_dim != self.v_head_dim:
            xv = ops.pad(xv, [0, self.head_dim - self.v_head_dim])
        return xv

    def unpad_attn_output(
        self, attn_output: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor | ReplicatedTensor:
        if self.head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]
        return attn_output

    def pre_process_attention(
        self,
        x: torch.Tensor | ReplicatedTensor,
        embedding: CachedRotaryLayer,
        start_positions: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor | ReplicatedTensor,
        torch.Tensor | ReplicatedTensor,
        torch.Tensor | ReplicatedTensor,
    ]:
        return self.latent_attn(x, embedding=embedding, start_positions=start_positions)


def create_paged_llama_attention_block(
    theta: Theta,
    *,
    config: LlamaModelConfig,
    block_index: int,
    head_count: int,
    head_dim: int,
    head_count_kv: int,
    rms_epsilon: float,
    model_arch: str,
    kv_cache: KVCache,
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
    sliding_window: Optional[int] = None,
    use_fused_qkv: bool = False,
):
    attn_type = attn_type_map[model_arch]

    block_class_map = {
        "gqa": PagedLlamaGQAttentionBlock,
        "mla": PagedLlamaMLAttentionBlock,
    }

    block_class = block_class_map.get(attn_type)
    if block_class is None:
        error_msg = f"Unsupported attention type to create PagedLlamaAttentionBlock: {attn_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return block_class(
        theta=theta,
        config=config,
        block_index=block_index,
        head_count=head_count,
        head_dim=head_dim,
        head_count_kv=head_count_kv,
        v_head_dim=v_head_dim,
        rms_epsilon=rms_epsilon,
        rope_dimension_count=rope_dimension_count,
        kv_cache=kv_cache,
        attention_kernel=attention_kernel,
        matmul_kernel=matmul_kernel,
        fake_quant=fake_quant,
        softcap=softcap,
        use_rope=use_rope,
        use_qk_norm=use_qk_norm,
        attn_temperature_tuning=attn_temperature_tuning,
        floor_scale=floor_scale,
        attention_scale=attention_scale,
        sliding_window=sliding_window,
        use_fused_qkv=use_fused_qkv,
    )
