# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from ...layers import *
from ...utils.create_cache import *
from ...types import Theta

torch.set_printoptions(profile="full")

__all__ = [
    "PagedGrokModelV1",
]

################################################################################
# Models
################################################################################


class PagedGrokModelV1(BaseCausalLMModel):
    """Grok model with a paged KV cache and supporting variable sequence
    length batched inference.

    As both the caching and batching setup is complicated, this model variant
    is modular, intending to be instantiated and used in an overall assembly
    vs trying to providing one-stop methods that do everything.

    The inference procedure is typically:

    1. Initialize the PagedKVCache state tensors.
    2. Generate an input mask given a vector of sequence lengths.
    3. Generate an attention mask from the input mask.
    4. Allocate a block mapping table.
    5. Invoke prefill() with a batch of sequences.
    6. Extract tokens from batched logits.
    7. Iteratively invoke decode() for as long as there are sequences needing
       to be serviced.

    Various samplers and schedulers can be interleaved throughout.
    """

    def __init__(self, theta: Theta, config: LlamaModelConfig):
        hp = config.hp
        super().__init__(
            theta,
            context_length=config.hp.context_length,
            device=config.device,
            activation_dtype=config.activation_dtype,
            attention_dtype=config.attention_dtype,
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
                device=self.device,
                use_hf=True,
                dtype=config.activation_dtype,
            ),
        )
        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module("output_lm_head", LinearLayer(theta("output")))

        self.attn_blocks = nn.ModuleList()
        self.moe_blocks = nn.ModuleList()

        for n in range(hp.block_count):
            self.attn_blocks.append(
                PagedLlamaAttentionBlock(
                    theta("blk", n),
                    block_index=n,
                    cache=self.cache,
                    head_count=hp.attention_head_count,
                    head_dim=hp.attn_head_dim,
                    attention_kernel=config.attention_kernel,
                    head_count_kv=hp.attention_head_count_kv,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                    softcap=hp.attention_softcap,
                )
            )
            self.moe_blocks.append(
                MoeBlock(
                    theta("blk", n),
                    expert_used_count=hp.expert_used_count,
                    rms_epsilon=hp.attention_layer_norm_rms_epsilon,
                    moe_activation=F.gelu,
                )
            )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: torch.Tensor,
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
        h *= math.sqrt(h.shape[-1])
        self.trace_tensor("grok.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, (attn_block, moe_block) in enumerate(
            zip(self.attn_blocks, self.moe_blocks)
        ):
            if block_idx == 0:
                self.trace_tensor(f"grok.attn_block.{block_idx}.input", h)

            h = attn_block(
                h,
                embedding=self.attention_embedding,
                start_index=0,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"grok.attn_block.{block_idx}.output", h)

            h = moe_block(h)
            self.trace_tensor(f"grok.moe_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        logits = logits / math.sqrt(3.0)
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
        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_mask = self.attention_embedding.compute_batch_mask(
            start_positions, batch_seq_len=1
        )
        self.trace_tensor("grok.embedding_batch_mask", embedding_batch_mask)

        h = self.token_embedding(tokens)
        h *= math.sqrt(h.shape[-1])
        self.trace_tensor("grok.token_embedding", h)

        # Iterate over attention blocks.
        for block_idx, (attn_block, moe_block) in enumerate(
            zip(self.attn_blocks, self.moe_blocks)
        ):
            if block_idx == 0:
                self.trace_tensor(f"grok.attn_block.{block_idx}.input", h)

            h = attn_block(
                h,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                embedding_batch_mask=embedding_batch_mask,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"grok.attn_block.{block_idx}.output", h)

            h = moe_block(h)
            self.trace_tensor(f"grok.moe_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)
        logits = logits / math.sqrt(3.0)
        return logits
