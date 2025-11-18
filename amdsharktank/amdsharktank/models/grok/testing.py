# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional
import torch

from amdsharktank.types.tensors import *
from amdsharktank.types.theta import Theta
from amdsharktank.utils.random import make_rand_torch
from amdsharktank.layers.testing import make_llama_attention_block_theta
from amdsharktank.layers.configs import LlamaModelConfig


def make_attention_block_ffn_theta_v2(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    expert_count: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=head_count,
        head_count_kv=head_count_kv,
        head_dim=head_dim,
        embedding_length=embedding_length,
        dtype=dtype,
        dtype_norm=dtype,  # TODO: Is this correct? Could not verify original model
    )
    moe_theta = make_moe_block_theta(
        block_idx=block_idx,
        feature_dim=embedding_length,
        ffn_dim=embedding_length,
        num_experts=expert_count,
    )
    res_dict = attention_theta.tree
    res_dict.update(moe_theta.tree)
    return Theta(res_dict)


def make_moe_block_theta(
    block_idx=0, feature_dim=1024, ffn_dim=6144, num_experts=8
) -> Theta:
    return Theta(
        {
            f"ffn_gate_inp.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_inp.weight",
                data=make_rand_torch((num_experts, ffn_dim)),
            ),
            f"ffn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_norm.weight", data=make_rand_torch((ffn_dim))
            ),
            f"layer_output_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.layer_output_norm.weight",
                data=make_rand_torch((ffn_dim)),
            ),
            f"ffn_gate_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_gate_exps.weight",
                data=make_rand_torch((num_experts, feature_dim * num_experts, ffn_dim)),
            ),
            f"ffn_up_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_up_exps.weight",
                data=make_rand_torch((num_experts, feature_dim * num_experts, ffn_dim)),
            ),
            f"ffn_down_exps.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_down_exps.weight",
                data=make_rand_torch((num_experts, ffn_dim, feature_dim * num_experts)),
            ),
        }
    )


def make_random_grok_theta(
    config: LlamaModelConfig, vocab_size: int, dtype: Optional[torch.dtype] = None
) -> Theta:
    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
        )
    }
    for i in range(config.hp.block_count):
        res[f"blk.{i}"] = make_attention_block_ffn_theta_v2(
            block_idx=i,
            head_count=config.hp.attention_head_count,
            head_count_kv=config.hp.attention_head_count_kv,
            head_dim=config.hp.attn_head_dim,
            embedding_length=config.hp.embedding_length,
            expert_count=config.hp.expert_count,
            dtype=dtype,
        ).tree

    res[f"output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
    )
    res[f"output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=make_rand_torch((1, config.hp.embedding_length), dtype=dtype),
    )

    return Theta(res)
