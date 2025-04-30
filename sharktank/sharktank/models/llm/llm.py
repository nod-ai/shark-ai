# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import math

import torch
import torch.nn as nn

from sharktank.layers import *
from sharktank.types import *
from sharktank.utils.create_cache import *

__all__ = [
    "PagedLlmModelV1",
    "AttentionFFNBlock",
]

################################################################################
# Models
################################################################################


class PagedLlmModelV1(BaseCausalLMModel):
    """Causal LLM Model with a paged KV cache and supporting variable sequence
    length batched inference.

    As both the caching and batching setup is complicated, this model variant
    is modular, intending to be instantiated and used in an overall assembly
    vs trying to providing one-stop methods that do everything.

    The inference procedure is typically:

    1. Initialize the kv cache state tensors.
    2. Generate an input mask given a vector of sequence lengths.
    3. Generate an attention mask from the input mask.
    4. Allocate a block mapping table.
    5. Invoke prefill() with a batch of sequences.
    6. Extract tokens from batched logits.
    7. Iteratively invoke decode() for as long as there are sequences needing
       to be serviced.

    Various samplers and schedulers can be interleaved throughout.

    In the case of tensor sharding (config.tensor_parallelism_size > 1) the model's KV
    cache head dimension is sharded.
    The number of KV cache heads must be divisible by the parallelism size.
    With this sharding approach the KV cache is not replicated across devices.
    The cache is split across the devices while the indexing logic/computation is
    replicated.
    All other arguments aside from the cache state are replicated.
    After the attention we all-reduce.
    The the first fully connected layer is split along the parallel dimension.
    This drives that the reduction dimension is split for the second FC layer.
    We return the unreduced tensor. The user is free to reduce it to obtain the
    unsharded result or chain it with other tensor-parallel operations.
    """

    def __init__(self, theta: Theta, config: LlamaModelConfig):
        super().__init__(
            theta,
            context_length=config.hp.context_length,
            device=config.device,
            activation_dtype=config.activation_dtype,
            attention_dtype=config.attention_dtype,
            fake_quant=config.fake_quant,
            static_tables=config.static_tables,
        )
        self.config = config
        self.hp = self.config.hp
        self.cache = create_paged_kv_cache(self.config)
        # TODO: Add inference_norm as an optional value from config
        self.inference_norm = self.config.hp.model_arch == "grok"

        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta("token_embd"), dtype=self.activation_dtype),
        )
        self.attention_embedding = nn.ModuleList(
            [
                RotaryEmbeddingLayer(
                    rope_dimension_count=self.hp.rope_dimension_count,
                    rope_freq_base=self.hp.rope_freq_base,
                    max_seqlen=self.hp.context_length,
                    device=self.device,
                    use_hf=self.config.use_hf,
                    tensor_parallelism_size=self.config.tensor_parallelism_size,
                    pipeline_parallelism=config.pipeline_parallelism_size > 1,
                    devices=self.cache.pipeline_to_device_lookup[pipeline],
                    dtype=self.config.activation_dtype,
                )
                for pipeline in range(self.config.pipeline_parallelism_size)
            ]
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
                    config=self.config,
                    fake_quant=self.fake_quant,
                    block_to_device_lookup=self.config.block_to_device_lookup,
                )
                for n in range(self.hp.block_count)
            ]
        )

    def _inter_layer_callback(self, x: ShardedTensor, curr_block: int) -> ShardedTensor:
        from ... import ops

        if curr_block == len(self.config.block_to_device_lookup) - 1:
            return x

        curr_devices = self.config.block_to_device_lookup[curr_block]
        next_devices = self.config.block_to_device_lookup[curr_block + 1]
        if all(d_curr == d_next for d_curr, d_next in zip(curr_devices, next_devices)):
            return x

        shards = ShardedTensor.move_shards_to_new_devices(
            x.shards, old_devices=curr_devices, new_devices=next_devices
        )
        return x.clone(ts=shards, devices=next_devices)

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: Union[torch.Tensor, ReplicatedTensor],
        *,
        # [[1, 1, batch_seq_len, batch_seq_len] x self.config.pipeline_parallelism_size]
        attention_mask: list[Union[torch.Tensor, ReplicatedTensor]],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: list[Union[torch.Tensor, ReplicatedTensor]],
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
    ):
        self._assert_device(tokens)
        self._assert_device(*attention_mask, dtype=self.activation_dtype)
        self._assert_device(*seq_block_ids)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # TODO: Get the normalization factor via configuration
        if self.inference_norm:
            h *= math.sqrt(h.shape[-1])

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(
                h,
                embedding=self.attention_embedding[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                start_index=0,
                attention_mask=attention_mask[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                cache_state=cache_state,
                seq_block_ids=seq_block_ids[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
            )
            h = self._inter_layer_callback(h, block_idx)
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)

        if self.inference_norm:
            logits = logits / math.sqrt(3.0)

        if "float8" in str(logits.dtype) or logits.dtype == torch.bfloat16:
            return logits.to(dtype=torch.float16)

        return logits

    def decode(
        self,
        # [bs, 1]
        tokens: Union[torch.Tensor, ReplicatedTensor],
        *,
        # [[bs, 1, 1, batch_seq_len] x self.config.pipeline_parallelism_size]
        attention_mask: list[Union[torch.Tensor, ReplicatedTensor]],
        # [bs] of starting positions
        start_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: list[Union[torch.Tensor, ReplicatedTensor]],
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
    ):
        assert len(tokens.shape) == 2
        assert all(len(mask.shape) == 4 for mask in attention_mask)
        assert all(len(start_position.shape) == 1 for start_position in start_positions)
        assert all(len(seq_block_id.shape) == 2 for seq_block_id in seq_block_ids)
        assert all(mask.shape[0] == tokens.shape[0] for mask in attention_mask)
        assert all(
            start_position.shape[0] == tokens.shape[0]
            for start_position in start_positions
        )
        assert all(
            seq_block_id.shape[0] == tokens.shape[0] for seq_block_id in seq_block_ids
        )
        assert tokens.shape[1] == 1
        assert all(mask.shape[1] == 1 and mask.shape[2] == 1 for mask in attention_mask)
        assert all(
            seq_block_ids[0].shape[1] == seq_block_id.shape[1]
            for seq_block_id in seq_block_ids[1:]
        )
        assert all(
            mask.shape[3] == seq_block_ids[0].shape[1] * self.config.block_seq_stride
            for mask in attention_mask
        )
        self._assert_device(tokens)
        self._assert_device(*attention_mask, dtype=self.activation_dtype)
        self._assert_device(*start_positions)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        # Precompute a position based mask for computing rope embeddings
        # as it is the same for all blocks.
        embedding_batch_masks = []
        for pipeline, start_position in enumerate(start_positions):
            mask = self.attention_embedding[pipeline].compute_batch_mask(
                start_position, batch_seq_len=1
            )
            embedding_batch_masks.append(mask)
            # TODO: How to name and trace this properly
            self.trace_tensor("llama.embedding_batch_mask", mask)

        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # TODO: Get the normalization factor via configuration
        if self.inference_norm:
            h *= math.sqrt(h.shape[-1])

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            h = block(  # TODO: Should we index into attention_mask and cache here?
                h,  # TODO: Hacky, shouldn't need to read info out of self.cache
                start_positions=start_positions[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                embedding=self.attention_embedding[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                embedding_batch_mask=embedding_batch_masks[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                attention_mask=attention_mask[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
                cache_state=cache_state,
                seq_block_ids=seq_block_ids[
                    self.cache.block_to_pipeline_lookup[block_idx]
                ],
            )
            h = self._inter_layer_callback(h, block_idx)
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = self.output_norm(h)
        logits = self.output_lm_head(h)

        if self.inference_norm:
            logits = logits / math.sqrt(3.0)

        if "float8" in str(logits.dtype) or logits.dtype == torch.bfloat16:
            return logits.to(dtype=torch.float16)

        return logits


################################################################################
# Layers
################################################################################


class AttentionFFNBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama using a
    paged cache."""

    def __init__(
        self,
        theta: Theta,
        *,
        block_index: int,
        cache: PagedAttention,  # TODO: Add deepseek PagedLatentAttention
        config: LlamaModelConfig,
        fake_quant: bool = True,
        block_to_device_lookup: tuple[tuple[int, ...], ...] | None = None,
    ):
        super().__init__(theta)

        attention_kernel = (
            "decomposed" if config.hp.model_arch == "grok" else config.attention_kernel
        )

        self.add_module(
            "attn",
            PagedLlamaAttentionBlock(
                theta=theta,
                block_index=block_index,
                cache=cache,
                head_count=config.hp.attention_head_count,
                head_dim=config.hp.attn_head_dim,
                head_count_kv=config.hp.attention_head_count_kv,
                rms_epsilon=config.hp.attention_layer_norm_rms_epsilon,
                attention_kernel=attention_kernel,
                fake_quant=fake_quant,
                block_to_device_lookup=block_to_device_lookup,
                softcap=config.hp.attention_softcap,
            ),
        )

        moe_func_map = {
            "llama": (
                torch.nn.functional.softmax,
                torch.nn.functional.silu,
                True,
                False,
            ),
            "grok": (
                torch.nn.functional.softmax,
                torch.nn.functional.gelu,
                True,
                False,
            ),
            "deepseek2": (
                torch.nn.functional.sigmoid,
                torch.nn.functional.silu,
                False,
                True,
            ),
        }

        if config.hp.expert_count:
            (
                score_experts,
                moe_activation,
                add_residual,
                normalize_experts,
            ) = moe_func_map[config.hp.model_arch]

            self.add_module(
                "ffn",
                MoeBlock(
                    theta=theta,
                    expert_used_count=config.hp.expert_used_count,
                    rms_epsilon=config.hp.attention_layer_norm_rms_epsilon,
                    moe_activation=moe_activation,
                    add_residual=add_residual,
                    score_experts=score_experts,
                    normalize_experts=normalize_experts,
                ),
            )
        else:
            self.add_module(
                "ffn",
                FFN(
                    theta=theta,
                    rms_epsilon=config.hp.attention_layer_norm_rms_epsilon,
                    fake_quant=fake_quant,
                ),
            )

    def forward(
        self,
        h: Union[torch.Tensor, ReplicatedTensor],
        *,
        embedding: RotaryEmbeddingLayer,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: list[Union[torch.Tensor, ReplicatedTensor]] = None,
        embedding_batch_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
    ):
        h = self.attn(
            h,
            embedding=embedding,
            seq_block_ids=seq_block_ids,
            start_index=start_index,
            start_positions=start_positions,
            attention_mask=attention_mask,
            embedding_batch_mask=embedding_batch_mask,
            cache_state=cache_state,
        )

        # Feed forward network.
        final_output = self.ffn(h)

        return final_output
