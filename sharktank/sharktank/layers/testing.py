# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from ..types.theta import Theta
from ..utils.testing import make_rand


def make_llama_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype | None = None,
    norm_dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": make_rand(
                name=f"blk.{block_idx}.attn_q.weight",
                shape=(head_count * head_dim, embedding_length),
                dtype=dtype,
            ),
            "attn_k.weight": make_rand(
                name=f"blk.{block_idx}.attn_k.weight",
                shape=(head_count_kv * head_dim, embedding_length),
                dtype=dtype,
            ),
            "attn_v.weight": make_rand(
                name=f"blk.{block_idx}.attn_v.weight",
                shape=(head_count_kv * head_dim, embedding_length),
                dtype=dtype,
            ),
            "attn_output.weight": make_rand(
                name=f"blk.{block_idx}.attn_output.weight",
                shape=(embedding_length, embedding_length),
                dtype=dtype,
            ),
            "attn_norm.weight": make_rand(
                name=f"blk.{block_idx}.attn_norm.weight",
                shape=(embedding_length),
                dtype=norm_dtype,
            ),
        }
    )


def make_latent_attention_block_theta(
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
    return Theta(
        {
            "wq.weight": make_rand(
                name=f"blk.{block_idx}.wq.weight",
                shape=(heads * (rope_dim + nope_dim), dim),
                dtype=dtype,
            ),
            "wkv_a.weight": make_rand(
                name=f"blk.{block_idx}.wkv_a.weight",
                shape=(kv_latent_dim + rope_dim, dim),
                dtype=dtype,
            ),
            "wkv_b.weight": make_rand(
                name=f"blk.{block_idx}.wkv_b.weight",
                shape=(heads * (v_head_dim + nope_dim), kv_latent_dim),
                dtype=dtype,
            ),
            "wo.weight": make_rand(
                name=f"blk.{block_idx}.wo.weight",
                shape=(dim, heads * v_head_dim),
                dtype=dtype,
            ),
            "attn_norm.weight": make_rand(
                name=f"blk.{block_idx}.attn_norm.weight",
                shape=(dim,),
                dtype=dtype,
            ),
            "kv_norm.weight": make_rand(
                name=f"blk.{block_idx}.kv_norm.weight",
                shape=(kv_latent_dim,),
                dtype=dtype,
            ),
        }
    )


def make_mmdit_double_block_random_theta(
    in_channels: int = 128,
    hidden_size: int = 3072,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    in_channels = 128
    hidden_size = 3072
    mlp_ratio = 4.0
    mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
    mlp_hidden_size2 = int(mlp_ratio * hidden_size)
    mlp_hidden_size3 = int(2 * (mlp_ratio - 1) * hidden_size)
    return Theta(
        {
            "img_attn.norm.key_norm.scale": make_rand(
                shape=(in_channels,), dtype=dtype
            ),
            "img_attn.norm.query_norm.scale": make_rand(
                shape=(in_channels,), dtype=dtype
            ),
            "img_attn.proj.bias": make_rand(shape=(hidden_size,), dtype=dtype),
            "img_attn.proj.weight": make_rand(
                shape=(hidden_size, hidden_size), dtype=dtype
            ),
            "img_attn.qkv.bias": make_rand(shape=(mlp_hidden_size,), dtype=dtype),
            "img_attn.qkv.weight": make_rand(
                shape=(mlp_hidden_size, hidden_size), dtype=dtype
            ),
            "img_mlp.0.bias": make_rand(shape=(mlp_hidden_size2), dtype=dtype),
            "img_mlp.0.weight": make_rand(
                shape=(mlp_hidden_size2, hidden_size), dtype=dtype
            ),
            "img_mlp.2.bias": make_rand(shape=(hidden_size), dtype=dtype),
            "img_mlp.2.weight": make_rand(
                shape=(hidden_size, mlp_hidden_size2), dtype=dtype
            ),
            "img_mod.lin.bias": make_rand(shape=(mlp_hidden_size3,), dtype=dtype),
            "img_mod.lin.weight": make_rand(
                shape=(mlp_hidden_size3, hidden_size), dtype=dtype
            ),
            "txt_attn.norm.key_norm.scale": make_rand(
                shape=(in_channels,), dtype=dtype
            ),
            "txt_attn.norm.query_norm.scale": make_rand(
                shape=(in_channels,), dtype=dtype
            ),
            "txt_attn.proj.bias": make_rand(shape=(hidden_size,), dtype=dtype),
            "txt_attn.proj.weight": make_rand(
                shape=(hidden_size, hidden_size), dtype=dtype
            ),
            "txt_attn.qkv.bias": make_rand(shape=(mlp_hidden_size,), dtype=dtype),
            "txt_attn.qkv.weight": make_rand(
                shape=(mlp_hidden_size, hidden_size), dtype=dtype
            ),
            "txt_mlp.0.bias": make_rand(shape=(mlp_hidden_size2), dtype=dtype),
            "txt_mlp.0.weight": make_rand(
                shape=(mlp_hidden_size2, hidden_size), dtype=dtype
            ),
            "txt_mlp.2.bias": make_rand(shape=(hidden_size), dtype=dtype),
            "txt_mlp.2.weight": make_rand(
                shape=(hidden_size, mlp_hidden_size2), dtype=dtype
            ),
            "txt_mod.lin.bias": make_rand(shape=(mlp_hidden_size3,), dtype=dtype),
            "txt_mod.lin.weight": make_rand(
                shape=(mlp_hidden_size3, hidden_size), dtype=dtype
            ),
        }
    )


def make_mmdit_single_block_random_theta(
    in_channels: int = 128,
    hidden_size: int = 3072,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
    mlp_hidden_size2 = int((mlp_ratio + 1) * hidden_size)
    mlp_hidden_size3 = int((2 * mlp_ratio - 1) * hidden_size)
    return Theta(
        {
            "norm.key_norm.scale": make_rand(shape=(in_channels,), dtype=dtype),
            "norm.query_norm.scale": make_rand(shape=(in_channels,), dtype=dtype),
            "attn.proj.bias": make_rand(shape=(hidden_size,), dtype=dtype),
            "attn.proj.weight": make_rand(
                shape=(hidden_size, hidden_size), dtype=dtype
            ),
            "linear1.bias": make_rand(shape=(mlp_hidden_size3,), dtype=dtype),
            "linear1.weight": make_rand(
                shape=(mlp_hidden_size3, hidden_size), dtype=dtype
            ),
            "linear2.bias": make_rand(shape=(hidden_size), dtype=dtype),
            "linear2.weight": make_rand(
                shape=(hidden_size, mlp_hidden_size2), dtype=dtype
            ),
            "modulation.lin.bias": make_rand(shape=(mlp_hidden_size,), dtype=dtype),
            "modulation.lin.weight": make_rand(
                shape=(mlp_hidden_size, hidden_size), dtype=dtype
            ),
        }
    )
