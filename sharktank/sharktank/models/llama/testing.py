# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import torch

from sharktank.types.tensors import *
from sharktank.types.theta import Theta
from sharktank.layers.configs import LlamaModelConfig
from sharktank.utils.testing import make_rand_torch
from sharktank.layers.testing import (
    make_llama_attention_block_theta,
    make_random_moe_block_theta,
)


def make_attention_block_theta(
    feature_dim: int,
    ffn_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
            "ffn_gate.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_up.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_down.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, ffn_dim), dtype=dtype)
            ),
            "ffn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
        }
    )


def make_attention_block_ffn_theta_v2(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    feed_forward_length: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=head_count,
        head_count_kv=head_count_kv,
        head_dim=head_dim,
        embedding_length=embedding_length,
        dtype=dtype,
    )
    ffn_theta = Theta(
        {
            "ffn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.ffn_norm.weight",
                data=make_rand_torch((head_count * head_dim), dtype=dtype),
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
    res_dict = attention_theta.tree
    res_dict.update(ffn_theta.tree)
    return Theta(res_dict)


def make_attention_moe_block_random_theta(
    block_idx: int, config: LlamaModelConfig, dtype: torch.dtype
) -> Theta:
    res_dict = {}
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=config.hp.attention_head_count,
        head_count_kv=config.hp.attention_head_count_kv,
        head_dim=config.hp.attn_head_dim,
        embedding_length=config.hp.embedding_length,
        dtype=dtype,
    )
    res_dict.update(attention_theta.tree)
    ffn_norm_theta = Theta(
        {
            "ffn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    (config.hp.attention_head_count * config.hp.attn_head_dim),
                    dtype=dtype,
                ),
            )
        }
    )
    res_dict.update(ffn_norm_theta.tree)
    moe_theta = make_random_moe_block_theta(
        in_dim=config.hp.embedding_length,
        expert_hidden_dim=config.hp.expert_feed_forward_length,
        num_experts=config.hp.expert_count,
        with_ffn_norm=False,
        num_shared_experts=config.hp.expert_shared_count,
        shared_expert_hidden_dim=config.hp.expert_shared_feed_forward_length,
        with_layer_output_norm=False,
        dtype=dtype,
    )
    res_dict.update(moe_theta.tree)
    return Theta(res_dict)


def make_random_llama_theta(
    config: LlamaModelConfig,
    vocab_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Theta:
    if vocab_size is None:
        vocab_size = config.vocabulary_size
    if dtype is None:
        dtype = config.dtype
    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
        )
    }
    for i in range(config.hp.block_count):
        is_moe_block = i in config.moe_layers
        if is_moe_block:
            # This is used in Llama 4.
            block = make_attention_moe_block_random_theta(
                config=config, block_idx=i, dtype=dtype
            ).tree
        else:
            block = make_attention_block_ffn_theta_v2(
                block_idx=i,
                head_count=config.hp.attention_head_count,
                head_count_kv=config.hp.attention_head_count_kv,
                head_dim=config.hp.attn_head_dim,
                embedding_length=config.hp.embedding_length,
                feed_forward_length=config.hp.feed_forward_length,
                dtype=dtype,
            ).tree
        res[f"blk.{i}"] = block

    res[f"output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=make_rand_torch((vocab_size, config.hp.embedding_length), dtype=dtype),
    )
    res[f"output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=make_rand_torch((1, config.hp.embedding_length), dtype=dtype),
    )

    return Theta(res)
