# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import torch

from sharktank.types.tensors import *
from sharktank.types.theta import Theta
from sharktank.utils.testing import make_rand_torch
from sharktank.layers.testing import make_latent_attention_block_theta
from sharktank.layers.configs.llm_configs import LlamaModelConfig


def make_deepseek_attention_block(
    *,
    block_idx: int,
    dim: int,
    heads: int,
    rope_dim: int,
    nope_dim: int,
    kv_latent_dim: int,
    v_head_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    attention_theta = make_latent_attention_block_theta(
        block_idx=block_idx,
        dim=dim,
        heads=heads,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        kv_latent_dim=kv_latent_dim,
        v_head_dim=v_head_dim,
        dtype=dtype,
    )
    moe_theta = make_moe_block_theta(block_idx=block_idx)
    res_dict = attention_theta.tree
    res_dict.update(moe_theta.tree)
    return Theta(res_dict)


def make_moe_block_theta(
    block_idx=0, expert_hidden_dim=16, ffn_dim=32, num_experts=4, shared_experts=1
) -> Theta:
    return Theta(
        {
            f"ffn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_norm.weight", data=make_rand_torch((ffn_dim))
            ),
            # Routed experts tensors
            f"ffn_gate_inp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_inp.weight",
                data=make_rand_torch((num_experts, ffn_dim)),
            ),
            f"ffn_gate_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_exps.weight",
                data=make_rand_torch((num_experts, expert_hidden_dim, ffn_dim)),
            ),
            f"ffn_up_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up_exps.weight",
                data=make_rand_torch((num_experts, expert_hidden_dim, ffn_dim)),
            ),
            f"ffn_down_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down_exps.weight",
                data=make_rand_torch((num_experts, ffn_dim, expert_hidden_dim)),
            ),
            # Shared experts tensors
            f"ffn_gate_shexp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_shexp.weight",
                data=make_rand_torch((shared_experts * expert_hidden_dim, ffn_dim)),
            ),
            f"ffn_up_shexp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up_shexp.weight",
                data=make_rand_torch((shared_experts * expert_hidden_dim, ffn_dim)),
            ),
            f"ffn_down_shexp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down_shexp.weight",
                data=make_rand_torch((ffn_dim, shared_experts * expert_hidden_dim)),
            ),
        }
    )


def make_random_deepseek_theta(
    config: LlamaModelConfig, vocab_size: int, dtype: Optional[torch.dtype] = None
) -> Theta:
    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
        )
    }
    for i in range(config.hp.block_count):
        res[f"blk.{i}"] = make_deepseek_attention_block(
            block_idx=i,
            dim=config.hp.embedding_length,
            heads=config.hp.attention_head_count,
            rope_dim=config.hp.rope_dimension_count,
            nope_dim=config.hp.nope_dim,
            kv_latent_dim=config.hp.kv_latent_dim,
            v_head_dim=config.hp.v_head_dim,
            dtype=dtype,
        ).tree

    res[f"output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
    )
    res[f"output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=make_rand_torch((config.hp.embedding_length), dtype=dtype),
    )

    return Theta(res)
