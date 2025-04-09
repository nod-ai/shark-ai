# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from ..types.theta import Theta
from ..types.tensors import DefaultPrimitiveTensor
from ..utils.testing import make_rand_torch


def make_llama_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype | None = None,
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
                data=make_rand_torch((embedding_length), dtype=dtype),
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
            "wq.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.wq.weight",
                data=make_rand_torch((heads * (rope_dim + nope_dim), dim), dtype=dtype),
            ),
            "wkv_a.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.wkv_a.weight",
                data=make_rand_torch((kv_latent_dim + rope_dim, dim), dtype=dtype),
            ),
            "wkv_b.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.wkv_b.weight",
                data=make_rand_torch(
                    (heads * (v_head_dim + nope_dim), kv_latent_dim), dtype=dtype
                ),
            ),
            "wo.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.wo.weight",
                data=make_rand_torch((dim, heads * v_head_dim), dtype=dtype),
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_norm.weight",
                data=make_rand_torch((dim,), dtype=dtype),
            ),
            "kv_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.kv_norm.weight",
                data=make_rand_torch((kv_latent_dim,), dtype=dtype),
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
            )
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
