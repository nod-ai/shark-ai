# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Toy GPT-OSS (OpenWeight) model generator for testing."""

from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.types import Dataset
from sharktank.models.llama.testing import make_random_llama_theta

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_gpt_oss.irpa")


def generate(seed: int) -> tuple:
    """Generate a toy GPT-OSS model for testing.

    Based on the real GPT-OSS config but scaled down:
    - Real: 36 layers, 128 experts, 2880 hidden_size, 201088 vocab
    - Toy: 4 layers, 8 experts, 128 hidden_size, 256 vocab
    """
    dtype = torch.float16
    block_seq_stride = 16
    max_blocks = 8

    # GPT-OSS specific parameters (scaled down for testing)
    attention_head_count = 8  # Real: 64
    attention_head_count_kv = 2  # Real: 8
    attn_head_dim = 16  # Real: 64
    vocabulary_size = 256  # Real: 201088
    expert_count = 8  # Real: 128
    experts_per_token = 2  # Real: 4
    hidden_size = attention_head_count * attn_head_dim  # 128 vs real 2880

    config = LlamaModelConfig(
        hp=LlamaHParams(
            model_arch="gpt-oss",
            vocab_size=vocabulary_size,
            context_length=block_seq_stride * max_blocks,
            embedding_length=hidden_size,
            block_count=4,  # Real: 36
            feed_forward_length=hidden_size,  # Real: 2880 (intermediate_size)
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=1e-5,
            attention_head_count_kv=attention_head_count_kv,
            rope_dimension_count=attn_head_dim,
            rope_freq_base=150000.0,  # Real GPT-OSS value
            # MoE configuration
            expert_count=expert_count,
            expert_used_count=experts_per_token,
            expert_feed_forward_length=hidden_size,
            # GPT-OSS specific configs (from get_custom_configs)
            is_moe_model=True,
            moe_block_type="PreGatherFFNMOE",
            use_moe_swiglu=True,
            sliding_window=32,  # Real: 128, scaled down
            swiglu_limit=7.0,
            rope_gpt_oss=True,
            use_fused_qkv=True,
            use_direct_expert_routing=True,
            use_residual_moe=True,
            use_ffn_norm=False,
            use_ffn_residual=False,
            moe_score_function="softmax",
            moe_activation_function="swiglu",
            normalize_moe_experts=False,
            # YaRN scaling (from real model)
            yarn_factor=32.0,  # rope_scaling_factor
            yarn_beta_slow=1.0,  # rope_ntk_alpha
            yarn_beta_fast=32.0,  # rope_ntk_beta
            yarn_original_context_len=4096,  # initial_context_length
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        attention_dtype=dtype,
        use_hf=False,  # Use sharktank native implementation
        dtype=dtype,
    )

    torch.manual_seed(seed)
    # Reuse llama theta generation since gpt-oss uses similar structure
    theta = make_random_llama_theta(
        config=config,
        vocab_size=vocabulary_size,
    )

    return theta, config


def main():
    args = parser.parse_args()
    theta, config = generate(args.seed)

    config_dict = config.hp.to_gguf_props()
    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)

    print(f"Generated toy GPT-OSS model: {args.output}")
    print(f"Config: {config.hp.block_count} layers, {config.hp.expert_count} experts")


if __name__ == "__main__":
    main()
