"""Toy GPT-OSS model generator for testing and development."""

from typing import Callable
from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.models.gpt_oss.testing import (
    make_random_gpt_oss_theta,
    make_simple_analytical_gpt_oss_theta,
    make_wide_range_weights,
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


def copy_weights_to_reference(shark_theta, ref_model, hp):
    if ref_model is None:
        return

    # Copy token embeddings
    ref_model.embedding.weight.data = shark_theta("token_embd", "weight").as_torch()

    # Copy transformer blocks
    for block_idx in range(hp.block_count):
        ref_block = ref_model.block[block_idx]
        ref_block.attn.norm.scale.data = (
            shark_theta("blk", block_idx, "attn_norm", "weight").as_torch().float()
        )
        ref_block.attn.qkv.weight.data = shark_theta(
            "blk", block_idx, "attn", "wqkv", "weight"
        ).as_torch()
        ref_block.attn.qkv.bias.data = shark_theta(
            "blk", block_idx, "attn", "wqkv", "bias"
        ).as_torch()
        ref_block.attn.out.weight.data = shark_theta(
            "blk", block_idx, "attn_output", "weight"
        ).as_torch()
        ref_block.attn.out.bias.data = shark_theta(
            "blk", block_idx, "attn_output", "bias"
        ).as_torch()
        ref_block.attn.sinks.data = shark_theta(
            "blk", block_idx, "attn_sinks"
        ).as_torch()

        ref_block.mlp.norm.scale.data = (
            shark_theta("blk", block_idx, "ffn_norm_scale", "weight").as_torch().float()
        )
        ref_block.mlp.gate.weight.data = shark_theta(
            "blk", block_idx, "ffn_gate_inp", "weight"
        ).as_torch()
        ref_block.mlp.gate.bias.data = shark_theta(
            "blk", block_idx, "ffn_gate_inp", "bias"
        ).as_torch()

        # Concatenate gate and up weights for SwiGLU
        gate_exps_weight = shark_theta(
            "blk", block_idx, "ffn_gate_exps", "weight"
        ).as_torch()
        gate_exps_bias = shark_theta(
            "blk", block_idx, "ffn_gate_exps", "bias"
        ).as_torch()
        up_exps_weight = shark_theta(
            "blk", block_idx, "ffn_up_exps", "weight"
        ).as_torch()
        up_exps_bias = shark_theta("blk", block_idx, "ffn_up_exps", "bias").as_torch()

        num_experts = gate_exps_weight.shape[0]
        intermediate_size = gate_exps_weight.shape[1]
        hidden_size = gate_exps_weight.shape[2]

        mlp1_weight = torch.zeros(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            dtype=gate_exps_weight.dtype,
            device=gate_exps_weight.device,
        )
        mlp1_bias = torch.zeros(
            num_experts,
            intermediate_size * 2,
            dtype=gate_exps_bias.dtype,
            device=gate_exps_bias.device,
        )

        mlp1_weight[:, :intermediate_size, :] = gate_exps_weight
        mlp1_weight[:, intermediate_size:, :] = up_exps_weight
        mlp1_bias[:, :intermediate_size] = gate_exps_bias
        mlp1_bias[:, intermediate_size:] = up_exps_bias

        ref_block.mlp.mlp1_weight.data = mlp1_weight
        ref_block.mlp.mlp1_bias.data = mlp1_bias

        ref_block.mlp.mlp2_weight.data = shark_theta(
            "blk", block_idx, "ffn_down_exps", "weight"
        ).as_torch()
        ref_block.mlp.mlp2_bias.data = shark_theta(
            "blk", block_idx, "ffn_down_exps", "bias"
        ).as_torch()

    # Copy output layers
    ref_model.norm.scale.data = shark_theta("output_norm", "weight").as_torch().float()
    ref_model.unembedding.weight.data = shark_theta("output", "weight").as_torch()


def calculate_cross_entropy_manual(
    model_instance, sequence: list[int], use_prefill: bool = True
) -> tuple[float, float]:
    """Calculate cross entropy and perplexity manually for debugging."""
    evaluator = model_instance.make_perplexity_eval()
    if use_prefill:
        res = evaluator.prefill_cross_entropy([sequence])[0]
    else:
        res = evaluator.decode_cross_entropy([sequence])[0]

    assert res.valid
    ce = res.score
    ppl = float(torch.exp(torch.tensor(ce)))

    print("cross_entropy_nats:", ce)
    print("perplexity:", ppl)
    return ce, ppl


def generate(
    seed: int,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
):
    """Generate a minimal deterministic GPT-OSS model for testing."""
    torch.manual_seed(seed)

    # Model architecture parameters
    block_seq_stride = 16
    max_blocks = 8
    attention_head_count = 8
    attn_head_dim = 32
    attention_head_count_kv = 4
    rope_dimension_count = 32
    vocabulary_size = 128
    block_count = 3
    feed_forward_length = 64

    # MoE parameters
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
    seed: int,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
):
    """Generate a minimal analytical GPT-OSS model with simple weights for hand calculation.

    This creates a tiny GPT-OSS model with all key features enabled but scaled down:
    - Simple weights (0, 1, -1, 0.5, 2) for hand calculation
    """
    torch.manual_seed(seed)

    # Minimal model architecture for analytical testing
    block_seq_stride = 4
    max_blocks = 2
    attention_head_count = 2
    attn_head_dim = 2
    attention_head_count_kv = 1
    rope_dimension_count = 2
    vocabulary_size = 8
    block_count = 2
    feed_forward_length = 4

    # Minimal MoE configuration
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
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            expert_feed_forward_length=expert_feed_forward_length,
            sliding_window=4,
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

    # Convert to GGUF format and save
    config_dict = config.hp.to_gguf_props()
    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
