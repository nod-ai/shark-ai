# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import PathLike
from collections import OrderedDict

import torch

from .wan import WanConfig, WanModel
from .export import export_wan_transformer, wan_transformer_default_batch_sizes
from sharktank.types import DefaultPrimitiveTensor, Theta
from sharktank.utils.random import make_rand_torch


def convert_wan_transformer_input_for_hugging_face_model(
    x: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("hidden_states", x),
            ("encoder_hidden_states", context),
            ("timestep", t),
        ]
    )


def make_wan_attn_block_random_theta(
    cross_attn_type="t2v_cross_attn",
    dim: int = 5120,
    ffn_dim: int = 13824,
    dtype: torch.dtype | None = None,
) -> Theta:
    mod_dim = 6
    if cross_attn_type == "t2v_cross_attn":
        return Theta(
            {
                "cross_attn.k.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.k.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "cross_attn.norm_k.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.norm_q.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.o.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.o.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "cross_attn.q.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.q.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "cross_attn.v.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "cross_attn.v.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "ffn.0.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((ffn_dim,), dtype=dtype)
                ),
                "ffn.0.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((ffn_dim, dim), dtype=dtype)
                ),
                "ffn.2.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "ffn.2.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, ffn_dim), dtype=dtype)
                ),
                "modulation": DefaultPrimitiveTensor(
                    data=make_rand_torch((1, mod_dim, dim), dtype=dtype)
                ),
                "norm3.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=torch.float32)
                ),
                "norm3.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=torch.float32)
                ),
                "self_attn.k.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.k.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "self_attn.norm_k.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.norm_q.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.o.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.o.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "self_attn.q.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.q.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
                "self_attn.v.bias": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim,), dtype=dtype)
                ),
                "self_attn.v.weight": DefaultPrimitiveTensor(
                    data=make_rand_torch((dim, dim), dtype=dtype)
                ),
            }
        )
    else:
        raise NotImplementedError


def make_random_head_theta(config: WanConfig, dtype: torch.dtype):
    head_dim = 64
    tensor_dict = {
        "head.modulation": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    1,
                    2,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.weight": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    head_dim,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((head_dim,), dtype=dtype)
        ),
    }
    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_random_ffnemb_theta(config: WanConfig, dtype: torch.dtype):
    dim = config.dim
    ffn_dim = config.ffn_dim
    tensor_dict = {
        "ffn.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((ffn_dim,), dtype=dtype)
        ),
        "ffn.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((ffn_dim, dim), dtype=dtype)
        ),
        "ffn.2.bias": DefaultPrimitiveTensor(data=make_rand_torch((dim,), dtype=dtype)),
        "ffn.2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((dim, ffn_dim), dtype=dtype)
        ),
    }
    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_random_time_guidance_projector_theta(config: WanConfig, dtype: torch.dtype):
    time_proj_dim = 6
    tensor_dict = {
        "time_projection.1.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((time_proj_dim * config.dim, config.dim), dtype=dtype)
        ),
        "time_projection.1.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((time_proj_dim * config.dim,), dtype=dtype)
        ),
    }
    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_random_head_theta(config: WanConfig, dtype: torch.dtype):
    head_dim = 64
    tensor_dict = {
        "head.modulation": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    1,
                    2,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.weight": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    head_dim,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((head_dim,), dtype=dtype)
        ),
    }
    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_random_mlpproj_theta(config: WanConfig, dtype: torch.dtype):
    dim = config.dim
    img_embed_dim = dim // 4
    tensor_dict = {
        "img_emb.proj.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((img_embed_dim,), dtype=dtype)
        ),
        "img_emb.proj.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((img_embed_dim,), dtype=dtype)
        ),
        "img_emb.proj.1.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((img_embed_dim, img_embed_dim), dtype=dtype)
        ),
        "img_emb.proj.1.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((img_embed_dim,), dtype=dtype)
        ),
        "img_emb.proj.3.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((dim, img_embed_dim), dtype=dtype)
        ),
        "img_emb.proj.3.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((dim,), dtype=dtype)
        ),
        "img_emb.proj.4.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((dim,), dtype=dtype)
        ),
        "img_emb.proj.4.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((dim,), dtype=dtype)
        ),
    }
    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_random_theta(config: WanConfig, dtype: torch.dtype):
    time_proj_dim = 6
    head_dim = 64
    tensor_dict = {
        "patch_embedding.weight": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (config.dim, config.out_dim, *config.patch_size), dtype=dtype
            )
        ),
        "patch_embedding.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim,), dtype=dtype)
        ),
        "time_embedding.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim, config.freq_dim), dtype=dtype)
        ),
        "time_embedding.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim,), dtype=dtype)
        ),
        "time_embedding.2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim, config.dim), dtype=dtype)
        ),
        "time_embedding.2.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim,), dtype=dtype)
        ),
        "time_projection.1.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((time_proj_dim * config.dim, config.dim), dtype=dtype)
        ),
        "time_projection.1.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((time_proj_dim * config.dim,), dtype=dtype)
        ),
        "text_embedding.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim, config.text_dim), dtype=dtype)
        ),
        "text_embedding.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim,), dtype=dtype)
        ),
        "text_embedding.2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim, config.dim), dtype=dtype)
        ),
        "text_embedding.2.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((config.dim,), dtype=dtype)
        ),
        "head.modulation": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    1,
                    2,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.weight": DefaultPrimitiveTensor(
            data=make_rand_torch(
                (
                    head_dim,
                    config.dim,
                ),
                dtype=dtype,
            )
        ),
        "head.head.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((head_dim,), dtype=dtype)
        ),
    }

    for i in range(config.num_layers):
        tensor_dict[f"blocks.{i}"] = make_wan_attn_block_random_theta(
            cross_attn_type="t2v_cross_attn",
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            dtype=dtype,
        ).flatten()

    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_t2v_single_layer_config():
    return WanConfig(num_heads=1, num_layers=1)


def make_toy_config() -> WanConfig:
    return WanConfig(dim=512, num_heads=5, num_layers=5)


def export_wan_random_single_layer(
    dtype: torch.dtype,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
):
    rng_state = torch.get_rng_state()
    torch.random.manual_seed(12345)

    dtype = torch.bfloat16
    params = make_t2v_single_layer_config()
    theta = make_random_theta(params, dtype)
    wan = WanModel(
        theta=theta,
        params=params,
    )

    export_wan_transformer(
        wan,
        mlir_output_path=mlir_output_path,
        parameters_output_path=parameters_output_path,
        batch_sizes=batch_sizes,
    )

    torch.set_rng_state(rng_state)
