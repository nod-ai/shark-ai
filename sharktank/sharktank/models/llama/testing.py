# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import torch

from ...types.tensors import *
from ...types.theta import Theta
from typing import Optional
from .llama import LlamaModelConfig
import torch
from ...utils.testing import make_rand
from ...layers.testing import make_llama_attention_block_theta


def make_attention_block_theta(
    feature_dim: int,
    ffn_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": make_rand(shape=(feature_dim, feature_dim), dtype=dtype),
            "attn_k.weight": make_rand(shape=(feature_dim, feature_dim), dtype=dtype),
            "attn_v.weight": make_rand(shape=(feature_dim, feature_dim), dtype=dtype),
            "attn_output.weight": make_rand(
                shape=(feature_dim, feature_dim), dtype=dtype
            ),
            "attn_norm.weight": make_rand(shape=(feature_dim), dtype=dtype),
            "ffn_gate.weight": make_rand(shape=(ffn_dim, feature_dim), dtype=dtype),
            "ffn_up.weight": make_rand(shape=(ffn_dim, feature_dim), dtype=dtype),
            "ffn_down.weight": make_rand(shape=(feature_dim, ffn_dim), dtype=dtype),
            "ffn_norm.weight": make_rand(shape=(feature_dim), dtype=dtype),
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
    norm_dtype: torch.dtype | None = None,
) -> Theta:
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=head_count,
        head_count_kv=head_count_kv,
        head_dim=head_dim,
        embedding_length=embedding_length,
        dtype=dtype,
        norm_dtype=norm_dtype,
    )
    ffn_theta = Theta(
        {
            "ffn_norm.weight": make_rand(
                name=f"blk.{block_idx}.ffn_norm.weight",
                shape=(head_count * head_dim),
                dtype=norm_dtype,
            ),
            "ffn_gate.weight": make_rand(
                name=f"blk.{block_idx}.ffn_gate.weight",
                shape=(feed_forward_length, embedding_length),
                dtype=dtype,
            ),
            "ffn_up.weight": make_rand(
                name=f"blk.{block_idx}.ffn_up.weight",
                shape=(feed_forward_length, embedding_length),
                dtype=dtype,
            ),
            "ffn_down.weight": make_rand(
                name=f"blk.{block_idx}.ffn_down.weight",
                shape=(embedding_length, feed_forward_length),
                dtype=dtype,
            ),
        }
    )
    res_dict = attention_theta.tree
    res_dict.update(ffn_theta.tree)
    return Theta(res_dict)


def make_moe_block_theta(feature_dim=1024, ffn_dim=6144, num_experts=8) -> Theta:
    return Theta(
        {
            "blk.0.ffn_gate_inp.weight": make_rand(
                name="blk.0.ffn_gate_inp.weight",
                shape=(num_experts, ffn_dim),
            ),
            "blk.0.ffn_norm.weight": make_rand(
                name="blk.0.ffn_norm.weight", shape=(ffn_dim)
            ),
            "blk.0.layer_output_norm.weight": make_rand(
                name="blk.0.layer_output_norm.weight", shape=(ffn_dim)
            ),
            "blk.0.ffn_gate_exps.weight": make_rand(
                name="blk.0.layer_output_norm.weight",
                shape=(num_experts, feature_dim * num_experts, ffn_dim),
            ),
            "blk.0.ffn_up_exps.weight": make_rand(
                name="blk.0.ffn_up_exps.weight",
                shape=(num_experts, feature_dim * num_experts, ffn_dim),
            ),
            "blk.0.ffn_down_exps.weight": make_rand(
                name="blk.0.ffn_down_exps.weight",
                shape=(num_experts, ffn_dim, feature_dim * num_experts),
            ),
        }
    )


def make_random_llama_theta(
    config: LlamaModelConfig,
    vocab_size: int,
    dtype: Optional[torch.dtype] = None,
    norm_dtype: Optional[torch.dtype] = None,
) -> Theta:
    res = {
        "token_embd.weight": make_rand(
            name="token_embd.weight",
            shape=(vocab_size, config.hp.embedding_length),
            dtype=dtype,
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
            norm_dtype=norm_dtype,
        ).tree

    res[f"output.weight"] = make_rand(
        name="output.weight",
        shape=(vocab_size, config.hp.embedding_length),
        dtype=dtype,
    )
    res[f"output_norm.weight"] = make_rand(
        name="output_norm.weight",
        shape=(1, config.hp.embedding_length),
        dtype=norm_dtype,
    )

    return Theta(res)
