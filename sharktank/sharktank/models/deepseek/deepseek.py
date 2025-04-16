# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from ...layers import *
from ...types import *
from ...utils.create_cache import *


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
                device=self.device,
                dtype=config.activation_dtype,
                use_hf=config.use_hf,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))
        # hp.block_count = 1
        blk_range = range(0, hp.block_count)
        blk_range = [3]
        self.attn_blocks = nn.ModuleList(
            [
                AttentionFFNBlock(
                    theta("blk", n),
                    block_index=n,
                    cache=self.cache,
                    head_count=hp.attention_head_count,
                    head_dim=hp.attn_head_dim,
                    head_count_kv=hp.attention_head_count_kv,
                    v_head_dim=hp.v_head_dim,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                    rope_dimension_count=hp.rope_dimension_count,
                    expert_used_count=hp.expert_used_count,
                    expert_count=hp.expert_count,
                    n_expert_groups=hp.n_expert_groups,
                    n_limited_groups=hp.n_limited_groups,
                    n_dense_layers=hp.n_dense_layers,
                    route_scale=hp.route_scale,
                    model_arch=hp.model_arch,
                )
                for n in blk_range
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: Union[torch.Tensor, ReplicatedTensor],
        *,
        # [1, 1, batch_seq_len, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(tokens)
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(seq_block_ids)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        h = self.token_embedding(tokens)

        # Iterate over attention blocks.
        for _, block in enumerate(self.attn_blocks):
            h = block(
                h,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits

    def decode(
        self,
        # [bs, 1]
        tokens: torch.Tensor,
        *,
        # [bs, 1, 1, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs] of starting positions
        start_positions: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(tokens)
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(start_positions)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        bs, _ = tokens.shape
        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )

        h = self.token_embedding(tokens)

        # Iterate over attention blocks.
        for _, block in enumerate(self.attn_blocks):
            h = block(
                h,
                embedding=self.attention_embedding,
                start_positions=start_positions,
                attention_mask=attention_mask,
                embedding_batch_mask=embedding_batch_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        return logits


################################################################################
# Layers
################################################################################


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
        v_head_dim: int,
        rms_epsilon: float,
        rope_dimension_count: int,
        expert_used_count: int,
        expert_count: int,
        n_expert_groups: int,
        n_limited_groups: int,
        n_dense_layers: int,
        route_scale: Optional[float],
        model_arch: str,
    ):
        super().__init__(theta)
        self.add_module(
            "attn",
            PagedLlamaAttentionBlock(
                theta=theta,
                block_index=block_index,
                cache=cache,
                head_count=head_count,
                head_dim=head_dim,
                head_count_kv=head_count_kv,
                v_head_dim=v_head_dim,
                rms_epsilon=rms_epsilon,
                rope_dimension_count=rope_dimension_count,
                model_arch=model_arch,
            ),
        )

        func_map = {
            "llama": (F.softmax, False),
            "grok": (F.softmax, False),
            "deepseek2": (F.sigmoid, True),
        }

        score_experts, normalize_experts = func_map[model_arch]

        # print('theta.keys', theta.keys)

        if block_index >= n_dense_layers:

            self.add_module(
                "ffn",
                MoeBlock(
                    theta=theta,
                    rms_epsilon=rms_epsilon,
                    expert_count=expert_count,
                    expert_used_count=expert_used_count,
                    n_expert_groups=n_expert_groups,
                    n_limited_groups=n_limited_groups,
                    add_residual=False,
                    route_scale=route_scale,
                    score_experts=score_experts,
                    normalize_experts=normalize_experts,
                ),
            )
        else:
            self.add_module("ffn", FFN(theta=theta, rms_epsilon=rms_epsilon))

    def forward(
        self,
        h: Union[torch.Tensor, ReplicatedTensor],
        *,
        embedding: RotaryEmbeddingLayer,
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
    ):
        h = self.attn(
            h,
            embedding=embedding,
            start_index=start_index,
            start_positions=start_positions,
            attention_mask=attention_mask,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            embedding_batch_mask=embedding_batch_mask,
        )

        # Feed forward network.
        final_output = self.ffn(h)

        return final_output
