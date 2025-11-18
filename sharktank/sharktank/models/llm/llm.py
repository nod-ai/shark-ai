# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import math

import torch
import torch.nn as nn

from amdsharktank import ops
from amdsharktank.layers import *
from amdsharktank.layers.paged_attention import build_cache_from_config
from amdsharktank.types import *
from amdsharktank.types.pipelining import transfer_between_blocks
from amdsharktank.utils.create_cache import *
from amdsharktank.utils.attention import *
from amdsharktank import ops

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
        )
        self.config = config
        self.hp = self.config.hp
        # TODO: Add inference_norm as an optional value from config
        self.inference_norm = self.config.hp.model_arch == "grok"

        self.cache = build_cache_from_config(config)

        self.add_module(
            "token_embedding",
            TokenEmbeddingLayer(theta("token_embd"), dtype=self.activation_dtype),
        )
        self.attention_embedding = build_rotary_layer(
            rope_dimension_count=self.hp.rope_dimension_count,
            rope_freq_base=self.hp.rope_freq_base,
            interleave=self.config.hp.rope_interleave_emb,
            device=self.device,
            dtype=self.config.activation_dtype,
            yarn_beta_slow=self.hp.yarn_beta_slow,
            yarn_beta_fast=self.hp.yarn_beta_fast,
            yarn_factor=self.hp.yarn_factor,
            yarn_original_context_len=self.hp.yarn_original_context_len,
            pipeline_stage_to_device_map=self.config.pipeline_to_device_map,
        )

        self.add_module(
            "output_norm",
            RMSNormLayer(
                theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
            ),
        )
        self.add_module(
            "output_lm_head",
            LinearLayer(
                theta("output"),
                matmul_kernel=self.config.matmul_kernel,
            ),
        )
        self.attn_blocks = nn.ModuleList(
            [
                AttentionFFNBlock(
                    theta("blk", n),
                    block_index=n,
                    config=self.config,
                    kv_cache=self.cache,
                    fake_quant=self.fake_quant,
                )
                for n in range(self.hp.block_count)
            ]
        )

    def prefill(
        self,
        # [bs, batch_seq_len]
        tokens: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: CacheAllocation,
        start_positions: Optional[torch.Tensor] = None,
    ):
        tokens = transfer_between_blocks(
            tokens, curr_block_tensors=self.theta.tensor("blk", 0)
        )
        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # TODO: Get the normalization factor via configuration
        if self.inference_norm:
            h *= math.sqrt(h.shape[-1])

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)

            (h, start_positions, seq_lens, seq_block_ids) = transfer_between_blocks(
                h,
                start_positions,
                seq_lens,
                seq_block_ids,
                curr_block_tensors=self.theta.tensor("blk", block_idx),
            )
            h = block(
                h,
                embedding=self.attention_embedding,
                start_positions=start_positions,
                seq_lens=seq_lens,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = h.to(self.config.activation_dtype)
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
        tokens: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        # [bs] of starting positions
        start_positions: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: CacheAllocation,
    ):
        tokens = transfer_between_blocks(
            tokens, curr_block_tensors=self.theta.tensor("blk", 0)
        )
        h = self.token_embedding(tokens)
        self.trace_tensor("llama.token_embedding", h)

        # TODO: Get the normalization factor via configuration
        if self.inference_norm:
            h *= math.sqrt(h.shape[-1])

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            if block_idx == 0:
                self.trace_tensor(f"llama.attn_block.{block_idx}.input", h)
            (h, start_positions, seq_lens, seq_block_ids) = transfer_between_blocks(
                h,
                start_positions,
                seq_lens,
                seq_block_ids,
                curr_block_tensors=self.theta.tensor("blk", block_idx),
            )

            h = block(
                h,
                start_positions=start_positions,
                embedding=self.attention_embedding,
                seq_lens=seq_lens,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        h = h.to(self.config.activation_dtype)
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
        config: LlamaModelConfig,
        kv_cache: KVCache,
        fake_quant: bool = True,
    ):
        super().__init__(theta)

        attention_kernel = (
            "decomposed"
            if config.hp.model_arch in ["grok", "gpt-oss"]
            else config.attention_kernel
        )

        if config.hp.model_arch == "llama4":
            use_rope = (
                block_index in config.rope_layers if config.rope_layers else False
            )
        else:
            use_rope = True

        use_qk_norm = (
            block_index in config.rope_layers and config.use_qk_norm
            if config.rope_layers
            else False
        )

        # Only apply sliding window on even blocks when configured
        sliding_window = (
            config.hp.sliding_window
            if (block_index % 2 == 0 and config.hp.sliding_window > 0)
            else None
        )

        self.add_module(
            "attn",
            create_paged_llama_attention_block(
                theta=theta,
                config=config,
                block_index=block_index,
                head_count=config.hp.attention_head_count,
                head_dim=config.hp.attn_head_dim,
                head_count_kv=config.hp.attention_head_count_kv,
                v_head_dim=config.hp.v_head_dim,
                rms_epsilon=config.hp.attention_layer_norm_rms_epsilon,
                rope_dimension_count=config.hp.rope_dimension_count,
                attention_kernel=attention_kernel,
                matmul_kernel=config.matmul_kernel,
                fake_quant=fake_quant,
                softcap=config.hp.attention_softcap,
                model_arch=config.hp.model_arch,
                use_rope=use_rope,
                use_qk_norm=use_qk_norm,
                attn_temperature_tuning=config.hp.attn_temperature_tuning,
                floor_scale=config.hp.floor_scale,
                attention_scale=config.hp.attention_scale,
                kv_cache=kv_cache,
                sliding_window=sliding_window,
                use_fused_qkv=config.hp.use_fused_qkv,
            ),
        )

        # Add FFN norm
        self.ffn_norm = torch.nn.Identity()
        if theta.optional_tensor("ffn_norm") is not None:
            self.ffn_norm = RMSNormLayer(
                theta("ffn_norm"), epsilon=config.hp.attention_layer_norm_rms_epsilon
            )

        moe_func_map = {
            "llama": (
                ops.softmax,
                torch.nn.functional.silu,
                True,
                False,
            ),
            "grok": (
                ops.softmax,
                torch.nn.functional.gelu,
                True,
                False,
            ),
            "deepseek2": (
                ops.sigmoid,
                torch.nn.functional.silu,
                True,
                True,
            ),
            "llama4": (
                torch.nn.functional.sigmoid,
                torch.nn.functional.silu,
                True,
                False,
            ),
            "gpt-oss": (
                ops.softmax,
                lambda x, alpha=1.702, limit=config.hp.swiglu_limit: ops.swiglu(
                    x, alpha=alpha, limit=limit
                ),
                False,
                False,
            ),
        }

        (
            score_experts,
            moe_activation,
            self.add_residual,
            normalize_experts,
        ) = moe_func_map[config.hp.model_arch]

        is_moe_block = False
        experts_ffn_moe_block = "DenseFFNMOE"
        if config.hp.model_arch == "llama4":
            is_moe_block = block_index in config.moe_layers
            experts_ffn_moe_block = "PreGatherFFNMOE"

        if config.hp.model_arch == "gpt-oss":
            is_moe_block = config.hp.expert_count and config.hp.expert_used_count
            experts_ffn_moe_block = config.hp.moe_block_type

        n_dense_layers = config.hp.n_dense_layers
        if (
            n_dense_layers is not None and block_index >= n_dense_layers
        ) or is_moe_block:
            self.add_module(
                "ffn",
                MoeBlock(
                    theta=theta,
                    expert_count=config.hp.expert_count,
                    expert_used_count=config.hp.expert_used_count,
                    expert_shared_count=config.hp.expert_shared_count,
                    rms_epsilon=config.hp.attention_layer_norm_rms_epsilon,
                    n_expert_groups=config.hp.n_expert_groups,
                    n_limited_groups=config.hp.n_limited_groups,
                    route_scale=config.hp.route_scale,
                    moe_activation=moe_activation,
                    experts_ffn_moe_block=experts_ffn_moe_block,
                    score_experts=score_experts,
                    normalize_experts=normalize_experts,
                    model_arch=config.hp.model_arch,
                    topk_then_softmax=config.hp.topk_then_softmax,
                    use_residual_moe=config.hp.use_residual_moe,
                    use_moe_swiglu=config.hp.use_moe_swiglu,
                ),
            )
        else:
            self.add_module(
                "ffn",
                FFN(
                    theta=theta,
                    fake_quant=fake_quant,
                    matmul_kernel=config.matmul_kernel,
                ),
            )

    def forward(
        self,
        h: Union[torch.Tensor, ReplicatedTensor],
        *,
        embedding: CachedRotaryLayer,
        seq_lens: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_positions: Optional[torch.Tensor] = None,
        cache_state: CacheAllocation | None = None,
    ):
        h = self.attn(
            h,
            embedding=embedding,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            start_positions=start_positions,
            cache_state=cache_state,
        )
        # Feed forward network with config-driven behavior

        ffn_input = self.ffn_norm(h)

        # Feed forward network.
        final_output = self.ffn(ffn_input)

        if self.add_residual:
            final_output = h + final_output

        return final_output
