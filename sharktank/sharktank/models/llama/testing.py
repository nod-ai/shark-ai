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
from sharktank.layers.testing import make_llama_attention_block_theta


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


def make_moe_block_theta(feature_dim=1024, ffn_dim=6144, num_experts=8) -> Theta:
    return Theta(
        {
            "blk.0.ffn_gate_inp.weight": DefaultPrimitiveTensor(
                name="blk.0.ffn_gate_inp.weight",
                data=make_rand_torch((num_experts, ffn_dim)),
            ),
            "blk.0.ffn_norm.weight": DefaultPrimitiveTensor(
                name="blk.0.ffn_norm.weight", data=make_rand_torch((ffn_dim))
            ),
            "blk.0.layer_output_norm.weight": DefaultPrimitiveTensor(
                name="blk.0.layer_output_norm.weight", data=make_rand_torch((ffn_dim))
            ),
            "blk.0.ffn_gate_exps.weight": DefaultPrimitiveTensor(
                name="blk.0.layer_output_norm.weight",
                data=make_rand_torch((num_experts, feature_dim * num_experts, ffn_dim)),
            ),
            "blk.0.ffn_up_exps.weight": DefaultPrimitiveTensor(
                name="blk.0.ffn_up_exps.weight",
                data=make_rand_torch((num_experts, feature_dim * num_experts, ffn_dim)),
            ),
            "blk.0.ffn_down_exps.weight": DefaultPrimitiveTensor(
                name="blk.0.ffn_down_exps.weight",
                data=make_rand_torch((num_experts, ffn_dim, feature_dim * num_experts)),
            ),
        }
    )


def make_random_llama_theta(
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
            feed_forward_length=config.hp.feed_forward_length,
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
