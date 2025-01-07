# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from os import PathLike

from .flux import FluxParams, FluxModelV1
from .export import export_flux_transformer, flux_transformer_default_batch_sizes
from ...types import DefaultPrimitiveTensor, Theta, save_load_theta
from ...layers.testing import (
    make_rand_torch,
)


def make_random_theta(config: FluxParams, dtype: torch.dtype):
    # TODO: do not hardcode values.

    in_channels = config.in_channels
    in_channels2 = 128
    hidden_size = config.hidden_size
    mlp_ratio = config.mlp_ratio
    mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
    mlp_hidden_size2 = int(mlp_ratio * hidden_size)
    mlp_hidden_size3 = int(2 * (mlp_ratio - 1) * hidden_size)
    mlp_hidden_size4 = int((mlp_ratio + 1) * hidden_size)
    mlp_hidden_size5 = int((2 * mlp_ratio - 1) * hidden_size)
    context_in_dim = config.context_in_dim
    time_dim = 256
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
        "double_blocks.0.img_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.proj.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.proj.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.img_attn.qkv.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.qkv.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.img_mlp.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size2), dtype=dtype)
        ),
        "double_blocks.0.img_mlp.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size2, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.img_mlp.2.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size), dtype=dtype)
        ),
        "double_blocks.0.img_mlp.2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, mlp_hidden_size2), dtype=dtype)
        ),
        "double_blocks.0.img_mod.lin.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size3,), dtype=dtype)
        ),
        "double_blocks.0.img_mod.lin.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size3, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.proj.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.proj.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.qkv.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.qkv.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.txt_mlp.0.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size2), dtype=dtype)
        ),
        "double_blocks.0.txt_mlp.0.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size2, hidden_size), dtype=dtype)
        ),
        "double_blocks.0.txt_mlp.2.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size), dtype=dtype)
        ),
        "double_blocks.0.txt_mlp.2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, mlp_hidden_size2), dtype=dtype)
        ),
        "double_blocks.0.txt_mod.lin.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size3,), dtype=dtype)
        ),
        "double_blocks.0.txt_mod.lin.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size3, hidden_size), dtype=dtype)
        ),
        "single_blocks.0.norm.key_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "single_blocks.0.norm.query_norm.scale": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "single_blocks.0.attn.proj.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "single_blocks.0.attn.proj.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "single_blocks.0.linear1.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size5,), dtype=dtype)
        ),
        "single_blocks.0.linear1.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size5, hidden_size), dtype=dtype)
        ),
        "single_blocks.0.linear2.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size), dtype=dtype)
        ),
        "single_blocks.0.linear2.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((hidden_size, mlp_hidden_size4), dtype=dtype)
        ),
        "single_blocks.0.modulation.lin.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
        ),
        "single_blocks.0.modulation.lin.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
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

    return Theta(tensor_dict)


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
