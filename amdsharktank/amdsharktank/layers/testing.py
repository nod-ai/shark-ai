# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from amdsharktank.types.theta import Theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor, unbox_tensor
from amdsharktank.utils.random import make_rand_torch
from amdsharktank.types.sharding import *


def make_llama_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype = torch.float32,
    dtype_norm: torch.dtype = torch.float32,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_q.weight",
                data=make_rand_torch(
                    (head_count * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_k.weight",
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_v.weight",
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_output.weight",
                data=make_rand_torch((embedding_length, embedding_length), dtype=dtype),
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_norm.weight",
                data=make_rand_torch((embedding_length), dtype=dtype_norm),
            ),
        }
    )


def make_latent_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    embedding_length: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    kv_latent_dim: int,
    q_lora_rank: int,
    v_head_dim: int,
    dtype: torch.dtype,
    dtype_norm: torch.dtype,
) -> Theta:
    return Theta(
        {
            "attn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_norm.weight",
                data=make_rand_torch((embedding_length,), dtype=dtype_norm),
            ),
            "attn_q_a_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_q_a_norm.weight",
                data=make_rand_torch((q_lora_rank,), dtype=dtype_norm),
            ),
            "attn_kv_a_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_kv_a_norm.weight",
                data=make_rand_torch((kv_latent_dim,), dtype=dtype_norm),
            ),
            "attn_q_a.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_q_a.weight",
                data=make_rand_torch((q_lora_rank, embedding_length), dtype=dtype),
            ),
            "attn_q_b.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_q_b.weight",
                data=make_rand_torch(
                    (head_count * (qk_rope_head_dim + qk_nope_head_dim), q_lora_rank),
                    dtype=dtype,
                ),
            ),
            "attn_kv_a_mqa.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_kv_a_mqa.weight",
                data=make_rand_torch(
                    (kv_latent_dim + qk_rope_head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_kv_b.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_kv_b.weight",
                data=make_rand_torch(
                    (head_count_kv * (v_head_dim + qk_nope_head_dim), kv_latent_dim),
                    dtype=dtype,
                ),
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_output.weight",
                data=make_rand_torch(
                    (embedding_length, head_count * v_head_dim), dtype=dtype
                ),
            ),
        }
    )


def make_ffn_block_theta(
    *,
    block_idx: int,
    embedding_length: int,
    feed_forward_length: int,
    dtype_norm: torch.dtype,
    dtype: torch.dtype,
) -> Theta:
    return Theta(
        {
            "ffn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_norm.weight",
                data=make_rand_torch((embedding_length), dtype=dtype_norm),
            ),
            "ffn_gate.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate.weight",
                data=make_rand_torch(
                    (feed_forward_length, embedding_length), dtype=dtype
                ),
            ),
            "ffn_up.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up.weight",
                data=make_rand_torch(
                    (feed_forward_length, embedding_length), dtype=dtype
                ),
            ),
            "ffn_down.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down.weight",
                data=make_rand_torch(
                    (embedding_length, feed_forward_length), dtype=dtype
                ),
            ),
        }
    )


def make_mmdit_double_block_random_theta(
    hidden_size: int = 3072,
    num_heads: int = 24,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    head_dim = hidden_size // num_heads
    mlp_hidden_size = int(mlp_ratio * hidden_size)
    qkv_out_size = 3 * hidden_size
    modulation_size = hidden_size * 6
    return Theta(
        {
            "img_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "img_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "img_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "img_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "img_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((qkv_out_size,), dtype=dtype)
            ),
            "img_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((qkv_out_size, hidden_size), dtype=dtype)
            ),
            "img_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size), dtype=dtype)
            ),
            "img_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
            ),
            "img_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "img_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, mlp_hidden_size), dtype=dtype)
            ),
            "img_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size,), dtype=dtype)
            ),
            "img_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size, hidden_size), dtype=dtype)
            ),
            "txt_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "txt_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "txt_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "txt_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "txt_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((qkv_out_size,), dtype=dtype)
            ),
            "txt_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((qkv_out_size, hidden_size), dtype=dtype)
            ),
            "txt_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size), dtype=dtype)
            ),
            "txt_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
            ),
            "txt_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "txt_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, mlp_hidden_size), dtype=dtype)
            ),
            "txt_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size,), dtype=dtype)
            ),
            "txt_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size, hidden_size), dtype=dtype)
            ),
        }
    )


def make_mmdit_single_block_random_theta(
    hidden_size: int = 3072,
    num_heads: int = 24,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    head_dim = hidden_size // num_heads
    modulation_size = 3 * hidden_size
    linear1_hidden_size = hidden_size * 3 + mlp_hidden_dim
    return Theta(
        {
            "norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((head_dim,), dtype=dtype)
            ),
            "attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "linear1.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((linear1_hidden_size,), dtype=dtype)
            ),
            "linear1.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((linear1_hidden_size, hidden_size), dtype=dtype)
            ),
            "linear2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "linear2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (hidden_size, hidden_size + mlp_hidden_dim), dtype=dtype
                )
            ),
            "modulation.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size,), dtype=dtype)
            ),
            "modulation.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((modulation_size, hidden_size), dtype=dtype)
            ),
        }
    )


def make_random_ffn_theta(
    block_idx: int,
    in_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    out_dim: int | None = None,
    suffix: str = "",
):
    ffn_gate = "ffn_gate" + suffix
    ffn_up = "ffn_up" + suffix
    ffn_down = "ffn_down" + suffix
    if out_dim is None:
        out_dim = in_dim
    return Theta(
        {
            f"{ffn_gate}.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.{ffn_gate}.weight",
                data=make_rand_torch((hidden_dim, in_dim), dtype=dtype),
            ),
            f"{ffn_up}.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.{ffn_up}.weight",
                data=make_rand_torch((hidden_dim, in_dim), dtype=dtype),
            ),
            f"{ffn_down}.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.{ffn_down}.weight",
                data=make_rand_torch((out_dim, hidden_dim), dtype=dtype),
            ),
        }
    )


def make_random_moe_block_theta(
    *,
    block_idx: int,
    in_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    with_ffn_norm: bool = True,
    num_shared_experts: int = 0,
    with_layer_output_norm: bool = False,
    dtype_rest: torch.dtype,
    dtype_norm: torch.dtype,
    dtype_bias: Optional[torch.dtype] | None = None,
    with_bias: Optional[bool] = False,
    with_norm_scale: Optional[bool] = False,
) -> Theta:
    """Create theta for standard MoE block (weights only, no biases).

    Builds the common MoE structure with optional FFN norm, shared experts, layer output norm, and bias terms.
    """
    res = {}

    if with_ffn_norm:
        res["ffn_norm.weight"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_norm.weight",
            data=make_rand_torch((in_dim), dtype=dtype_norm),
        )

    res["ffn_gate_inp.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.ffn_gate_inp.weight",
        data=make_rand_torch((num_experts, in_dim), dtype=dtype_norm),
    )
    res["ffn_gate_exps.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.ffn_gate_exps.weight",
        data=make_rand_torch(
            (num_experts, expert_hidden_dim, in_dim), dtype=dtype_rest
        ),
    )
    res["ffn_up_exps.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.ffn_up_exps.weight",
        data=make_rand_torch(
            (num_experts, expert_hidden_dim, in_dim), dtype=dtype_rest
        ),
    )
    res["ffn_down_exps.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.ffn_down_exps.weight",
        data=make_rand_torch(
            (num_experts, in_dim, expert_hidden_dim), dtype=dtype_rest
        ),
    )

    if with_bias:
        res["ffn_gate_inp.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_gate_inp.bias",
            data=make_rand_torch((num_experts,), dtype=dtype_bias),
        )
        res["ffn_gate_exps.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_gate_exps.bias",
            data=make_rand_torch((num_experts, expert_hidden_dim), dtype=dtype_bias),
        )
        res["ffn_up_exps.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_up_exps.bias",
            data=make_rand_torch((num_experts, expert_hidden_dim), dtype=dtype_bias),
        )
        res["ffn_down_exps.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_down_exps.bias",
            data=make_rand_torch((num_experts, in_dim), dtype=dtype_bias),
        )

    if with_norm_scale:
        res["ffn_norm_scale.weight"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.ffn_norm_scale.weight",
            data=make_rand_torch((in_dim,), dtype=dtype_norm),
        )

    if num_shared_experts > 0:
        shared_ffn_theta = make_random_ffn_theta(
            block_idx=block_idx,
            in_dim=in_dim,
            hidden_dim=expert_hidden_dim * num_shared_experts,
            out_dim=in_dim,
            dtype=dtype_rest,
            suffix="_shexp",
        )
        res.update(shared_ffn_theta.tree)

    if with_layer_output_norm:
        res["layer_output_norm.weight"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.layer_output_norm.weight",
            data=make_rand_torch((in_dim), dtype=dtype_norm),
        )

    return Theta(res)


def make_fused_qkv_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype,
    dtype_norm: torch.dtype,
    with_bias: bool = True,
    with_attention_sinks: bool = True,
    with_attention_norm: bool = True,
) -> Theta:
    """Create theta for attention block with fused QKV projection.

    Generates attention weights with a single fused QKV projection instead of
    separate Q/K/V projections. Supports optional bias terms and attention sinks.

    """
    q_size = head_count * head_dim
    k_size = head_count_kv * head_dim
    v_size = head_count_kv * head_dim
    qkv_size = q_size + k_size + v_size

    res = {}

    if with_attention_norm:
        res["attn_norm.weight"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.attn_norm.weight",
            data=make_rand_torch((embedding_length,), dtype=dtype_norm),
        )

    res["attn.wqkv.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.attn.wqkv.weight",
        data=make_rand_torch((qkv_size, embedding_length), dtype=dtype),
    )

    res["attn_output.weight"] = DefaultPrimitiveTensor(
        name=f"blk.{block_idx}.attn_output.weight",
        data=make_rand_torch((embedding_length, q_size), dtype=dtype),
    )

    if with_bias:
        res["attn.wqkv.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.attn.wqkv.bias",
            data=make_rand_torch((qkv_size,), dtype=dtype),
        )
        res["attn_output.bias"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.attn_output.bias",
            data=make_rand_torch((embedding_length,), dtype=dtype),
        )

    if with_attention_sinks:
        res["attn_sinks"] = DefaultPrimitiveTensor(
            name=f"blk.{block_idx}.attn_sinks",
            data=make_rand_torch((head_count,), dtype=dtype),
        )

    return Theta(res)
