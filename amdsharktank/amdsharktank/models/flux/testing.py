# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import PathLike
from collections import OrderedDict

import torch

from .flux import FluxParams, FluxModelV1
from .export import export_flux_transformer, flux_transformer_default_batch_sizes
from amdsharktank.types import DefaultPrimitiveTensor, Theta
from amdsharktank.layers.testing import (
    make_mmdit_double_block_random_theta,
    make_mmdit_single_block_random_theta,
)
from amdsharktank.utils.random import make_rand_torch


def convert_flux_transformer_input_for_hugging_face_model(
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: torch.Tensor,
    y: torch.Tensor,
    guidance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("hidden_states", img),
            ("encoder_hidden_states", txt),
            ("pooled_projections", y),
            ("timestep", timesteps),
            ("img_ids", img_ids.reshape(img_ids.shape[1:])),
            ("txt_ids", txt_ids.reshape(txt_ids.shape[1:])),
            ("guidance", guidance),
        ]
    )


def make_random_theta(config: FluxParams, dtype: torch.dtype):
    in_channels = config.in_channels
    hidden_size = config.hidden_size
    mlp_ratio = config.mlp_ratio
    context_in_dim = config.context_in_dim
    time_dim = config.time_dim
    vec_dim = config.vec_in_dim
    patch_size = 1
    out_channels = config.out_channels
    tensor_dict = {
        "img_in.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, in_channels), dtype=dtype)
        ),
        "img_in.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "txt_in.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, context_in_dim), dtype=dtype)
        ),
        "txt_in.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "time_in.in_layer.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, time_dim), dtype=dtype)
        ),
        "time_in.in_layer.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "time_in.out_layer.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "time_in.out_layer.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "vector_in.in_layer.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, vec_dim), dtype=dtype)
        ),
        "vector_in.in_layer.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "vector_in.out_layer.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "vector_in.out_layer.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "final_layer.linear.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch(
                (patch_size * patch_size * out_channels, hidden_size), dtype=dtype
            )
        ),
        "final_layer.linear.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((patch_size * patch_size * out_channels,), dtype=dtype)
        ),
        "final_layer.adaLN_modulation.1.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size * 2, hidden_size), dtype=dtype)
        ),
        "final_layer.adaLN_modulation.1.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size * 2,), dtype=dtype)
        ),
    }

    for i in range(config.depth):
        tensor_dict[f"double_blocks.{i}"] = make_mmdit_double_block_random_theta(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            num_heads=config.num_heads,
            dtype=dtype,
        ).flatten()

    for i in range(config.depth_single_blocks):
        tensor_dict[f"single_blocks.{i}"] = make_mmdit_single_block_random_theta(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            num_heads=config.num_heads,
            dtype=dtype,
        ).flatten()

    if config.guidance_embed:
        tensor_dict["guidance_in.in_layer.weight"] = DefaultPrimitiveTensor(  #
            data=make_rand_torch(
                (
                    hidden_size,
                    time_dim,
                ),
                dtype=dtype,
            )
        )
        tensor_dict["guidance_in.in_layer.bias"] = DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        )
        tensor_dict["guidance_in.out_layer.weight"] = DefaultPrimitiveTensor(  #
            data=make_rand_torch(
                (
                    hidden_size,
                    hidden_size,
                ),
                dtype=dtype,
            )
        )
        tensor_dict["guidance_in.out_layer.bias"] = DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        )

    res = Theta(tensor_dict)
    res.rename_tensors_to_paths()
    return res


def make_dev_single_layer_config():
    return FluxParams(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )


def make_toy_config() -> FluxParams:
    num_heads = 5
    mlp_ratio = 2
    axes_dim = [4 * 2, 4 * 3, 4 * 4]
    in_channels = sum(axes_dim)
    hidden_size = in_channels * num_heads
    vec_in_dim = hidden_size // mlp_ratio
    assert hidden_size == mlp_ratio * vec_in_dim
    output_img_height = 2 * in_channels // 4
    output_img_width = 3 * in_channels // 4
    return FluxParams(
        in_channels=in_channels,
        out_channels=in_channels,
        time_dim=13,
        vec_in_dim=vec_in_dim,
        context_in_dim=7,
        txt_context_length=11,
        hidden_size=hidden_size,
        mlp_ratio=float(mlp_ratio),
        num_heads=num_heads,
        depth=3,
        depth_single_blocks=2,
        axes_dim=axes_dim,
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
        output_img_height=output_img_height,
        output_img_width=output_img_width,
        output_img_channels=3,
    )


def export_dev_random_single_layer(
    dtype: torch.dtype,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    rng_state = torch.get_rng_state()
    torch.random.manual_seed(12345)

    dtype = torch.bfloat16
    params = make_dev_single_layer_config()
    theta = make_random_theta(params, dtype)
    flux = FluxModelV1(
        theta=theta,
        params=params,
    )

    export_flux_transformer(
        flux,
        mlir_output_path=mlir_output_path,
        parameters_output_path=parameters_output_path,
        batch_sizes=batch_sizes,
    )

    torch.set_rng_state(rng_state)
