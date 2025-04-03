# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

import torch

from sharktank.types.tensors import *
from sharktank.types.theta import Theta


def make_conv2d_layer_theta(
    in_channels: int,
    out_channels: int,
    kernel_height: int,
    kernel_width: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "weight": DefaultPrimitiveTensor(
                data=torch.rand(
                    out_channels,
                    in_channels,
                    kernel_height,
                    kernel_width,
                    dtype=dtype,
                )
            ),
            "bias": DefaultPrimitiveTensor(
                data=torch.rand(out_channels, dtype=dtype),
            ),
        }
    )


def make_resnet_block_2d_theta(
    in_channels: int,
    out_channels: List[int],
    kernel_height: int,
    kernel_width: int,
    input_time_emb_channels: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "norm1.weight": DefaultPrimitiveTensor(
                data=torch.rand(in_channels, dtype=dtype)
            ),
            "norm1.bias": DefaultPrimitiveTensor(
                data=torch.rand(in_channels, dtype=dtype)
            ),
            "conv1": make_conv2d_layer_theta(
                in_channels=in_channels,
                out_channels=out_channels[0],
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
            "norm2.weight": DefaultPrimitiveTensor(
                data=torch.rand(out_channels[0], dtype=dtype)
            ),
            "norm2.bias": DefaultPrimitiveTensor(
                data=torch.rand(out_channels[0], dtype=dtype)
            ),
            "conv2": make_conv2d_layer_theta(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
            "time_emb_proj.weight": DefaultPrimitiveTensor(
                data=torch.rand(out_channels[0], input_time_emb_channels, dtype=dtype),
            ),
            "time_emb_proj.bias": DefaultPrimitiveTensor(
                data=torch.rand(out_channels[0], dtype=dtype),
            ),
            "conv_shortcut": make_conv2d_layer_theta(
                in_channels=in_channels,
                out_channels=out_channels[1],
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
        }
    )


def make_up_down_sample_2d_theta(
    in_channels: int,
    out_channels: int,
    kernel_height: int,
    kernel_width: int,
    dtype: torch.dtype | None = None,
):
    return Theta(
        {
            "conv": make_conv2d_layer_theta(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
        }
    )


def make_up_down_block_2d_theta(
    channels: int,
    kernel_height: int,
    kernel_width: int,
    input_time_emb_channels: int,
    resnet_layers: int,
    is_up_block: bool,
    dtype: torch.dtype | None = None,
) -> Theta:
    res = dict()
    assert channels % 2 == 0
    for i in range(resnet_layers):
        res[f"resnets.{i}"] = make_resnet_block_2d_theta(
            in_channels=channels,
            out_channels=[channels, channels // 2],
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            input_time_emb_channels=input_time_emb_channels,
            dtype=dtype,
        ).tree
    path_name = "upsamplers" if is_up_block else "downsamplers"
    res[f"{path_name}.0"] = make_up_down_sample_2d_theta(
        in_channels=channels // 2,
        out_channels=channels // 2,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        dtype=dtype,
    ).tree
    return Theta(res)
