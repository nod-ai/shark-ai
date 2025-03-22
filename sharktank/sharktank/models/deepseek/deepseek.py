# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import *
from ...types import *
from ...utils.create_cache import *
from ... import ops

__all__ = [
    "PagedDeepseekModelV1",
]

################################################################################
# Models
################################################################################


class PagedDeepseekModelV1(BaseCausalLMModel):
    """DeepseekModel with a paged KV cache and supporting variable sequence"""

    def __init__(self, theta: Theta, config: LlamaModelConfig):
        hp = config.hp
        super().__init__(
            theta,
            context_length=config.hp.context_length,
            device=config.device,
            activation_dtype=config.activation_dtype,
            attention_dtype=config.attention_dtype,
            static_tables=config.static_tables,
        )
        self.config = config
        self.hp = hp
        self.cache = create_paged_kv_cache(self.config)
        self.activation_dtype = config.activation_dtype

        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta("token_embd"), dtype=config.activation_dtype),
        )
        self.add_module(
            "attention_embedding",
            RotaryEmbeddingLayer(
                rope_dimension_count=hp.rope_dimension_count,
                rope_freq_base=hp.rope_freq_base,
                max_seqlen=hp.context_length,
                tensor_parallelism_size=config.tensor_parallelism_size,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))
        self.attn_blocks = nn.ModuleList(
            [
                AttentionFFNBlock(
                    theta("blk", n),
                    block_index=n,
                    cache=self.cache,
                    head_count=hp.attention_head_count,
                    head_dim=hp.attn_head_dim,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                    rope_dimension_count=hp.rope_dimension_count,
                    route_scale=hp.route_scale,
                    score_func=hp.expert_score_func,
                )
                for n in range(hp.block_count)
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: Union[torch.Tensor, ReplicatedTensor],
    ):
        h = self.token_embedding(tokens)

        # Iterate over attention blocks.
        for _, block in enumerate(self.attn_blocks):
            h = block(h, embedding=self.attention_embedding)

        h = self.output_norm(h)
        h = self.output_lm_head(h)
        return h


################################################################################
# Layers
################################################################################


class PagedLatentAttentionBlock(ThetaLayer):
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
        rope_dimension_count: int,
        attention_scale: Optional[float] = None,
    ):
        super().__init__(theta)

        self.block_index = block_index
        self.cache = cache
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.attention_scale = attention_scale
        self.rope_dimension_count = rope_dimension_count

        self.add_module(
            "attn_norm", RMSNormLayer(theta("attn_norm"), epsilon=rms_epsilon)
        )
        self.add_module("kv_norm", RMSNormLayer(theta("kv_norm"), epsilon=rms_epsilon))

        self.wq = None
        if "wq" in theta:
            self.wq = LinearLayer(theta("wq"))
        else:
            self.wq_a = LinearLayer(theta("wq_a"))
            self.wq_b = LinearLayer(theta("wq_b"))
            self.q_norm = RMSNormLayer(theta("q_norm"), epsilon=rms_epsilon)

        self.add_module("wkv_a", LinearLayer(theta("wkv_a")))
        self.add_module("wkv_b", LinearLayer(theta("wkv_b")))
        self.add_module("wo", LinearLayer(theta("wo")))

    def forward(
        self,
        h: torch.Tensor,
        *,
        embedding: RotaryEmbeddingLayer,
    ):
        h = self.attn_norm(h)

        if self.wq is not None:
            q = self.wq(h).unflatten(2, (self.head_count, -1))
        else:
            q = self.wq_b(self.q_norm(self.wq_a(h)))
            q = q.unflatten(2, (self.head_count, -1))

        qk_nope_head_dim = q.shape[-1] - self.rope_dimension_count
        q_nope = q[:, :, :, :qk_nope_head_dim]
        q_rope = q[:, :, :, qk_nope_head_dim:]
        q_rope = embedding(xt=q_rope, start_index=0)
        q = torch.cat((q_nope, q_rope), dim=-1)

        kv = self.wkv_a(h)
        kv_nope_size = kv.shape[-1] - self.rope_dimension_count
        kv_nope = kv[:, :, :kv_nope_size]
        k_rope = kv[:, :, kv_nope_size:]
        k_rope = embedding(xt=k_rope.unsqueeze(2), start_index=0)

        ## We should restructure this to apply the wkv_b post attention.
        kv_norm = self.kv_norm(kv_nope)
        wkv_b = self.wkv_b(kv_norm).unflatten(2, (self.head_count, -1))

        k_nope = wkv_b[:, :, :, :qk_nope_head_dim]
        v = wkv_b[:, :, :, qk_nope_head_dim:]

        k_rope = ops.repeat(k_rope, (1, 1, k_nope.shape[2] // k_rope.shape[2], 1))
        k = ops.cat((k_nope, k_rope), dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = ops.scaled_dot_product_attention(
            q=q,  # [bs, ..., sl, dim]
            k=k,  # [bs, ..., sl, dim]
            v=v,  # [bs, ..., sl, dim]
            a=None,  # [bs, ..., sl, sl]
            is_causal=True,  # assumes causal masking when true
            scale=self.attention_scale,  # defaults to 1/sqrt(dim)
        )

        attn = attn.transpose(1, 2)
        return self.wo(attn.flatten(2))


class AttentionFFNBlock(ThetaLayer):
    """Implements a self attention layer in the style of Deepseek using a
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
        rope_dimension_count: int,
        route_scale: Optional[float],
        score_func: Optional[str],
    ):
        super().__init__(theta)
        self.add_module(
            "attn",
            PagedLatentAttentionBlock(
                theta=theta,
                block_index=block_index,
                cache=cache,
                head_count=head_count,
                head_dim=head_dim,
                head_count_kv=head_count_kv,
                rms_epsilon=rms_epsilon,
                rope_dimension_count=rope_dimension_count,
            ),
        )

        func_map = {
            "sigmoid": (torch.nn.functional.sigmoid, True),
            "softmax": (torch.nn.functional.softmax, False),
        }

        score_experts, normalize_experts = func_map[score_func]

        if "ffn" in theta:
            self.add_module("ffn", FFN(theta=theta("ffn")))
        else:
            self.add_module(
                "ffn",
                MoeBlock(
                    theta=theta("moe"),
                    rms_epsilon=1,
                    expert_used_count=1,
                    add_residual=False,
                    route_scale=route_scale,
                    score_experts=score_experts,
                    normalize_experts=normalize_experts,
                ),
            )

        self.add_module(
            "ffn_norm", RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        )

    def forward(
        self,
        h: Union[torch.Tensor, ReplicatedTensor],
        *,
        embedding: RotaryEmbeddingLayer,
    ):
        h = h + self.attn(h, embedding=embedding)

        # Feed forward network.
        ffn_input = self.ffn_norm(h)
        ffn_down = self.ffn(ffn_input)
        final_output = h + ffn_down

        return final_output
