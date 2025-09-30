from typing import Callable
from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.models.gpt_oss.testing import (
    make_random_gpt_oss_theta,
    make_simple_analytical_gpt_oss_theta,
    make_wide_range_weights,
    make_simple_calculable_weight_torch,
)
from sharktank.types import Dataset

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--seed", default=12345, help="Random seed for deterministic generation"
)
parser.add_argument(
    "-o", "--output", default="/tmp/toy_gpt_oss.irpa", help="Output file path"
)
parser.add_argument(
    "--analytical",
    action="store_true",
    help="Use analytical model with simple weights for hand calculation",
)


def generate(
    seed,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
):
    """Generate a minimal deterministic GPT-OSS model for testing."""
    torch.manual_seed(seed)

    block_seq_stride = 16
    max_blocks = 8
    attention_head_count = 8
    attn_head_dim = 32
    attention_head_count_kv = 4
    rope_dimension_count = 32
    vocabulary_size = 128
    block_count = 3
    feed_forward_length = 64

    expert_count = 4
    expert_used_count = 2
    expert_feed_forward_length = 32

    config = LlamaModelConfig(
        hp=LlamaHParams(
            model_arch="gpt-oss",
            vocab_size=vocabulary_size,
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=block_count,
            feed_forward_length=feed_forward_length,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=1e-5,
            attention_head_count_kv=attention_head_count_kv,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=150000.0,
            rope_interleave_emb=False,
            yarn_factor=32.0,
            yarn_beta_slow=1.0,
            yarn_beta_fast=32.0,
            yarn_original_context_len=4096,
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            expert_feed_forward_length=expert_feed_forward_length,
            sliding_window=128,
            swiglu_limit=7.0,
            use_base_frequency_scaling=True,
            use_fused_qkv=True,
            topk_then_softmax=True,
            use_residual_moe=True,
            moe_block_type="PreGatherFFNMOE",
            use_moe_swiglu=True,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype_rest,
        attention_dtype=dtype_rest,
    )

    theta = make_random_gpt_oss_theta(
        config=config,
        vocab_size=vocabulary_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
        weight_generator=weight_generator,
    )
    return theta, config


def generate_analytical(
    seed,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
):
    """Generate a minimal analytical GPT-OSS model with simple weights for hand calculation.

    This creates a tiny GPT-OSS model with all key features enabled but scaled down:
    - Fused QKV attention (GPT-OSS specific)
    - Sliding window attention (window size 4)
    - MoE with SwiGLU (PreGatherFFNMOE)
    - Residual MoE connections
    - YaRN RoPE scaling
    - TopK then softmax routing
    - Simple weights (0, 1, -1, 0.5, 2) for hand calculation
    """
    torch.manual_seed(seed)

    block_seq_stride = 4
    max_blocks = 2
    attention_head_count = 2
    attn_head_dim = 2
    attention_head_count_kv = 1
    rope_dimension_count = 2
    vocabulary_size = 8
    block_count = 2
    feed_forward_length = 4

    expert_count = 2
    expert_used_count = 1
    expert_feed_forward_length = 2

    config = LlamaModelConfig(
        hp=LlamaHParams(
            model_arch="gpt-oss",
            vocab_size=vocabulary_size,
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=block_count,
            feed_forward_length=feed_forward_length,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=1e-5,
            attention_head_count_kv=attention_head_count_kv,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=10000.0,
            rope_interleave_emb=False,
            yarn_factor=1.0,
            yarn_beta_slow=1.0,
            yarn_beta_fast=1.0,
            yarn_original_context_len=block_seq_stride * max_blocks,
            # MoE config
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            expert_feed_forward_length=expert_feed_forward_length,
            sliding_window=4,  # Small sliding window for hand calculation
            swiglu_limit=7.0,
            use_base_frequency_scaling=True,  # Keep GPT-OSS feature
            use_fused_qkv=True,  # Keep GPT-OSS fused QKV
            topk_then_softmax=True,  # Keep GPT-OSS routing
            use_residual_moe=True,  # Keep GPT-OSS residual MoE
            moe_block_type="PreGatherFFNMOE",  # Keep GPT-OSS MoE type
            use_moe_swiglu=True,  # Keep GPT-OSS SwiGLU with MoE
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype_rest,
        attention_dtype=dtype_rest,
    )

    theta = make_simple_analytical_gpt_oss_theta(
        config=config,
        vocab_size=vocabulary_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
    )
    return theta, config


def main():
    args = parser.parse_args()

    if args.analytical:
        print("Generating analytical GPT-OSS model with simple weights...")
        theta, config = generate_analytical(args.seed)
    else:
        print("Generating standard GPT-OSS model with wide-range weights...")
        theta, config = generate(args.seed)

    config_dict = config.hp.to_gguf_props()

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
