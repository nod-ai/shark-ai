# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine import aot
from pathlib import Path
from amdsharktank import ops
from amdsharktank.models.punet.layers import ResnetBlock2D
from amdsharktank.models.punet.sharding import ResnetBlock2DSplitOutputChannelsSharding
from amdsharktank.transforms.dataset import set_float_dtype
from amdsharktank.types.tensors import *
from amdsharktank.types.theta import Theta, Dataset
from amdsharktank.utils.export import export
from amdsharktank.utils.iree import flatten_for_iree_signature
from amdsharktank.utils.random import make_rand_torch
from typing import Any, List

import functools
import torch


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
                data=make_rand_torch(
                    shape=[
                        out_channels,
                        in_channels,
                        kernel_height,
                        kernel_width,
                    ],
                    dtype=dtype,
                )
            ),
            "bias": DefaultPrimitiveTensor(
                data=make_rand_torch(shape=[out_channels], dtype=dtype),
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
                data=make_rand_torch(shape=[in_channels], dtype=dtype)
            ),
            "norm1.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(shape=[in_channels], dtype=dtype)
            ),
            "conv1": make_conv2d_layer_theta(
                in_channels=in_channels,
                out_channels=out_channels[0],
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
            "norm2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(shape=[out_channels[0]], dtype=dtype)
            ),
            "norm2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(shape=[out_channels[0]], dtype=dtype)
            ),
            "conv2": make_conv2d_layer_theta(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                dtype=dtype,
            ).tree,
            "time_emb_proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch(
                    shape=[out_channels[0], input_time_emb_channels], dtype=dtype
                ),
            ),
            "time_emb_proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch(shape=[out_channels[0]], dtype=dtype),
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


def toy_size_sharded_resnet_block_config() -> dict[str, Any]:
    batch_size = 2
    return {
        "batch_size": 2,
        "in_channels": 6,
        "out_channels": [12, 8],
        "height": 11,
        "width": 13,
        "kernel_height": 5,
        "kernel_width": 5,
        "temb_channels": 8,
        "norm_groups": 2,
        "eps": 0.01,
        "shard_count": 2,
        "non_linearity": "relu",
        "dropout": 0.0,
        "output_scale_factor": None,
    }


def export_sharded_toy_resnet_block_iree_test_data(
    mlir_path: Path,
    parameters_path: Path,
    input_args_path: Path,
    expected_results_path: Path,
    target_dtype: torch.dtype,
    reference_dtype: torch.dtype = torch.float64,
    shard_count: int = 2,
):
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        config = toy_size_sharded_resnet_block_config()
        reference_theta = make_resnet_block_2d_theta(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_height=config["kernel_height"],
            kernel_width=config["kernel_width"],
            input_time_emb_channels=config["temb_channels"],
            dtype=reference_dtype,
        )
        reference_theta.rename_tensors_to_paths()

        reference_input_image = torch.rand(
            config["batch_size"],
            config["in_channels"],
            config["height"],
            config["width"],
            dtype=reference_dtype,
        )
        reference_input_time_emb = torch.rand(
            config["batch_size"], config["temb_channels"], dtype=reference_dtype
        )

        reference_resnet_block = ResnetBlock2D(
            theta=reference_theta,
            groups=config["norm_groups"],
            eps=config["eps"],
            non_linearity=config["non_linearity"],
            output_scale_factor=config["output_scale_factor"],
            dropout=config["dropout"],
            temb_channels=config["temb_channels"],
            time_embedding_norm="default",
        )

        unsharded_theta = reference_theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )
        sharding_spec = ResnetBlock2DSplitOutputChannelsSharding(
            shard_count=shard_count
        )
        sharded_theta = ops.reshard(unsharded_theta, sharding_spec)
        sharded_dataset = Dataset({}, sharded_theta)
        sharded_dataset.save(parameters_path)
        sharded_dataset = Dataset.load(parameters_path)
        sharded_theta = sharded_dataset.root_theta

        sharded_resnet_block = ResnetBlock2D(
            theta=sharded_theta,
            groups=config["norm_groups"],
            eps=config["eps"],
            non_linearity=config["non_linearity"],
            output_scale_factor=config["output_scale_factor"],
            dropout=config["dropout"],
            temb_channels=config["temb_channels"],
            time_embedding_norm="default",
        )

        input_image = reference_input_image.to(dtype=target_dtype)
        input_time_emb = reference_input_time_emb.to(dtype=target_dtype)
        sharded_input_image = ops.reshard_split(input_image, dim=1, count=shard_count)
        sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)

        fxb = aot.FxProgramsBuilder(sharded_resnet_block)
        args = (
            sharded_input_image,
            sharded_input_time_emb,
        )
        # Export handles device affinities of sharded tensor arguments automatically.
        export(
            ResnetBlock2D.forward,
            fx_builder=fxb,
            args=args,
            strict=False,
        )
        exported_resnet_block = aot.export(
            fxb,
        )
        exported_resnet_block.save_mlir(mlir_path)

        expected_result = reference_resnet_block(
            reference_input_image, reference_input_time_emb
        )
        expected_result = ops.reshard_split(expected_result, dim=1, count=shard_count)

        iree_args = flatten_for_iree_signature(args)
        iree_expected_results = flatten_for_iree_signature(expected_result)

        iree_args_thata = Theta(
            {
                f"{i}": DefaultPrimitiveTensor(name=f"{i}", data=iree_args[i])
                for i in range(len(iree_args))
            }
        )
        iree_expected_results_theta = Theta(
            {
                f"{i}": DefaultPrimitiveTensor(
                    name=f"{i}", data=iree_expected_results[i]
                )
                for i in range(len(iree_expected_results))
            }
        )

        Dataset(root_theta=iree_args_thata, properties={}).save(input_args_path)
        Dataset(root_theta=iree_expected_results_theta, properties={}).save(
            expected_results_path
        )
