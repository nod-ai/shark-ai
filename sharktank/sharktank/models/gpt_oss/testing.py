from typing import Optional, Callable
import torch


from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Theta
from sharktank.layers.configs import LlamaModelConfig
from sharktank.utils.random import make_rand_torch
from sharktank.layers.testing import make_random_moe_block_theta
from sharktank.models.llama.testing import (
    make_wide_range_weights,
    make_simple_calculable_weight_torch,
)


def make_gpt_oss_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype,
    dtype_norm: torch.dtype,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
) -> Theta:
    """Create theta for GPT-OSS attention block with fused QKV weights."""

    q_size = head_count * head_dim
    k_size = head_count_kv * head_dim
    v_size = head_count_kv * head_dim
    qkv_size = q_size + k_size + v_size

    return Theta(
        {
            "attn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_norm.weight",
                data=weight_generator((embedding_length,), dtype_norm),
            ),
            "attn.wqkv.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn.wqkv.weight",
                data=weight_generator((qkv_size, embedding_length), dtype),
            ),
            "attn.wqkv.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn.wqkv.bias",
                data=weight_generator((qkv_size,), dtype),
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_output.weight",
                data=weight_generator((embedding_length, head_count * head_dim), dtype),
            ),
            "attn_output.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_output.bias",
                data=weight_generator((embedding_length,), dtype),
            ),
            "attn_sinks": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_sinks",
                data=weight_generator((head_count,), dtype),
            ),
        }
    )


def make_gpt_oss_moe_block_theta(
    *,
    block_idx: int,
    embedding_length: int,
    expert_feed_forward_length: int,
    expert_count: int,
    dtype: torch.dtype,
    dtype_norm: torch.dtype,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
) -> Theta:
    """Create theta for GPT-OSS MoE block."""

    return Theta(
        {
            "ffn_gate_inp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_inp.weight",
                data=weight_generator((expert_count, embedding_length), dtype),
            ),
            "ffn_gate_inp.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_inp.bias",
                data=weight_generator((expert_count,), dtype),
            ),
            "ffn_gate_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_exps.weight",
                data=weight_generator(
                    (expert_count, expert_feed_forward_length, embedding_length), dtype
                ),
            ),
            "ffn_gate_exps.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_exps.bias",
                data=weight_generator(
                    (expert_count, expert_feed_forward_length), dtype
                ),
            ),
            "ffn_up_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up_exps.weight",
                data=weight_generator(
                    (expert_count, expert_feed_forward_length, embedding_length), dtype
                ),
            ),
            "ffn_up_exps.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up_exps.bias",
                data=weight_generator(
                    (expert_count, expert_feed_forward_length), dtype
                ),
            ),
            "ffn_down_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down_exps.weight",
                data=weight_generator(
                    (expert_count, embedding_length, expert_feed_forward_length), dtype
                ),
            ),
            "ffn_down_exps.bias": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down_exps.bias",
                data=weight_generator((expert_count, embedding_length), dtype),
            ),
            # Add norm scale weight for gpt-oss
            "ffn_norm_scale.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_norm_scale.weight",
                data=weight_generator((embedding_length,), dtype_norm),
            ),
        }
    )


def make_gpt_oss_attention_moe_block_theta(
    block_idx: int,
    config: LlamaModelConfig,
    dtype_rest: torch.dtype,
    dtype_norm: torch.dtype,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
) -> Theta:
    """Create combined attention + MoE block theta for GPT-OSS."""
    res_dict = {}

    # Attention part with fused QKV
    attention_theta = make_gpt_oss_attention_block_theta(
        block_idx=block_idx,
        head_count=config.hp.attention_head_count,
        head_count_kv=config.hp.attention_head_count_kv,
        head_dim=config.hp.attn_head_dim,
        embedding_length=config.hp.embedding_length,
        dtype=dtype_rest,
        dtype_norm=dtype_norm,
        weight_generator=weight_generator,
    )
    res_dict.update(attention_theta.tree)

    # MoE part
    moe_theta = make_gpt_oss_moe_block_theta(
        block_idx=block_idx,
        embedding_length=config.hp.embedding_length,
        expert_feed_forward_length=config.hp.expert_feed_forward_length,
        expert_count=config.hp.expert_count,
        dtype=dtype_rest,
        dtype_norm=dtype_norm,
        weight_generator=weight_generator,
    )
    res_dict.update(moe_theta.tree)

    return Theta(res_dict)


def make_random_gpt_oss_theta(
    config: LlamaModelConfig,
    vocab_size: Optional[int] = None,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
    weight_generator: Callable[
        [list[int], torch.dtype], torch.Tensor
    ] = make_wide_range_weights,
) -> Theta:
    """Generate a GPT-OSS theta with configurable weight generation."""
    if vocab_size is None:
        vocab_size = config.hp.vocab_size

    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=weight_generator((vocab_size, config.hp.embedding_length), dtype_rest),
        )
    }

    # Create blocks - all are MoE blocks for GPT-OSS
    for i in range(config.hp.block_count):
        block = make_gpt_oss_attention_moe_block_theta(
            config=config,
            block_idx=i,
            dtype_rest=dtype_rest,
            dtype_norm=dtype_norm,
            weight_generator=weight_generator,
        ).tree
        res[f"blk.{i}"] = block

    # Output layers
    res["output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=weight_generator((vocab_size, config.hp.embedding_length), dtype_rest),
    )
    res["output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=weight_generator((config.hp.embedding_length,), dtype_norm),
    )

    return Theta(res)


def make_simple_analytical_gpt_oss_theta(
    config: LlamaModelConfig,
    vocab_size: Optional[int] = None,
    dtype_rest: torch.dtype = torch.bfloat16,
    dtype_norm: torch.dtype = torch.bfloat16,
) -> Theta:
    """Generate a GPT-OSS theta with simple analytical weights for hand calculation."""
    return make_random_gpt_oss_theta(
        config=config,
        vocab_size=vocab_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
        weight_generator=make_simple_calculable_weight_torch,
    )
