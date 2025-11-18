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
from amdsharktank.layers.testing import (
    make_latent_attention_block_theta,
    make_ffn_block_theta,
    make_random_moe_block_theta,
)
from amdsharktank.layers.configs.llm_configs import LlamaModelConfig


def make_deepseek_attention_block(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    embedding_length: int,
    feed_forward_length: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    kv_latent_dim: int,
    q_lora_rank: int,
    v_head_dim: int,
    n_dense_layers: int,
    expert_count: int,
    expert_shared_count: int,
    moe_intermediate_size: int,
    dtype_rest: torch.dtype,
    dtype_norm: torch.dtype,
) -> Theta:
    attention_theta = make_latent_attention_block_theta(
        block_idx=block_idx,
        head_count=head_count,
        head_count_kv=head_count_kv,
        embedding_length=embedding_length,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_latent_dim=kv_latent_dim,
        q_lora_rank=q_lora_rank,
        v_head_dim=v_head_dim,
        dtype=dtype_rest,
        dtype_norm=dtype_norm,
    )

    if block_idx >= n_dense_layers:
        ffn_theta = make_random_moe_block_theta(
            block_idx=block_idx,
            in_dim=embedding_length,
            expert_hidden_dim=moe_intermediate_size,
            num_experts=expert_count,
            num_shared_experts=expert_shared_count,
            with_layer_output_norm=False,
            dtype_rest=dtype_rest,
            dtype_norm=dtype_norm,
        )
    else:
        ffn_theta = make_ffn_block_theta(
            block_idx=block_idx,
            embedding_length=embedding_length,
            feed_forward_length=feed_forward_length,
            dtype=dtype_rest,
            dtype_norm=dtype_norm,
        )
    res_dict = attention_theta.tree
    res_dict.update(ffn_theta.tree)
    return Theta(res_dict)


def make_random_deepseek_theta(
    config: LlamaModelConfig,
    vocab_size: int,
    dtype_norm: torch.dtype,
    dtype_rest: torch.dtype,
) -> Theta:
    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=make_rand_torch(
                (vocab_size, config.hp.embedding_length), dtype=dtype_rest
            ),
        )
    }
    for i in range(config.hp.block_count):
        res[f"blk.{i}"] = make_deepseek_attention_block(
            block_idx=i,
            head_count=config.hp.attention_head_count,
            head_count_kv=config.hp.attention_head_count_kv,
            embedding_length=config.hp.embedding_length,
            feed_forward_length=config.hp.feed_forward_length,
            q_lora_rank=config.hp.q_lora_rank,
            qk_rope_head_dim=config.hp.qk_rope_head_dim,
            qk_nope_head_dim=config.hp.qk_nope_head_dim,
            kv_latent_dim=config.hp.kv_lora_rank,
            v_head_dim=config.hp.v_head_dim,
            n_dense_layers=config.hp.n_dense_layers,
            expert_count=config.hp.expert_count,
            expert_shared_count=config.hp.expert_shared_count,
            moe_intermediate_size=config.hp.moe_intermediate_size,
            dtype_rest=dtype_rest,
            dtype_norm=dtype_norm,
        ).tree

    res[f"output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=make_rand_torch(
            (vocab_size, config.hp.embedding_length), dtype=dtype_rest
        ),
    )
    res[f"output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=make_rand_torch((config.hp.embedding_length), dtype=dtype_norm),
    )

    return Theta(res)
