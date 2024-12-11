# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from iree.turbine import aot
from sharktank.models.flux.flux import (
    FluxModelV1,
    FluxParams,
)
import sharktank.ops as ops
from sharktank.layers.testing import (
    make_mmdit_double_block_random_theta,
    make_mmdit_single_block_random_theta,
)
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Theta


def make_rand_torch(shape: list[int], dtype: torch.dtype | None = torch.float32):
    return torch.rand(shape, dtype=dtype) * 2 - 1


# TODO: Refactor this to a function that generates random toy weights, possibly
# to another file
dtype = torch.float32
in_channels = 64
in_channels2 = 128
hidden_size = 3072
mlp_ratio = 4.0
mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
mlp_hidden_size2 = int(mlp_ratio * hidden_size)
mlp_hidden_size3 = int(2 * (mlp_ratio - 1) * hidden_size)
mlp_hidden_size4 = int((mlp_ratio + 1) * hidden_size)
mlp_hidden_size5 = int((2 * mlp_ratio - 1) * hidden_size)
context_in_dim = 4096
time_dim = 256
vec_dim = 768
patch_size = 1
out_channels = 64

flux_theta = Theta(
    {
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
        "time_in.0.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, time_dim), dtype=dtype)
        ),
        "time_in.0.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "time_in.1.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "time_in.1.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "vector_in.0.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, vec_dim), dtype=dtype)
        ),
        "vector_in.0.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "vector_in.1.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
        ),
        "vector_in.1.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.norm.key_norm.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.img_attn.norm.query_norm.weight": DefaultPrimitiveTensor(  #
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
        "double_blocks.0.txt_attn.norm.key_norm.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "double_blocks.0.txt_attn.norm.query_norm.weight": DefaultPrimitiveTensor(  #
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
        "single_blocks.0.attn.norm.key_norm.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((in_channels2,), dtype=dtype)
        ),
        "single_blocks.0.attn.norm.query_norm.weight": DefaultPrimitiveTensor(  #
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
        "single_blocks.0.mod.lin.bias": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
        ),
        "single_blocks.0.mod.lin.weight": DefaultPrimitiveTensor(
            data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
        ),
        "last_layer.outlinear.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch(
                (patch_size * patch_size * out_channels, hidden_size), dtype=dtype
            )
        ),
        "last_layer.outlinear.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((patch_size * patch_size * out_channels,), dtype=dtype)
        ),
        "last_layer.ada_linear.weight": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size * 2, hidden_size), dtype=dtype)
        ),
        "last_layer.ada_linear.bias": DefaultPrimitiveTensor(  #
            data=make_rand_torch((hidden_size * 2,), dtype=dtype)
        ),
    }
)


class FluxTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24
        self.batch_size = 5

    def testExport(self):
        params = FluxParams(
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
            guidance_embed=False,
        )
        theta = flux_theta
        flux = FluxModelV1(
            theta=theta,
            params=params,
        )

        img = torch.rand([self.batch_size, 1024, 64])
        img_ids = torch.rand([self.batch_size, 1024, 3])
        txt = torch.rand([self.batch_size, 512, 4096])
        txt_ids = torch.rand([self.batch_size, 512, 3])
        timesteps = torch.rand([self.batch_size])
        y = torch.rand([self.batch_size, 768])

        flux.forward(img, img_ids, txt, txt_ids, timesteps, y)
        fxb = aot.FxProgramsBuilder(flux)

        @fxb.export_program(
            name="flux", args=(img, img_ids, txt, txt_ids, timesteps, y), strict=False
        )
        def _(model, img, img_ids, txt, txt_ids, timesteps, y) -> torch.Tensor:
            return model.forward(img, img_ids, txt, txt_ids, timesteps, y)

        output = aot.export(fxb)
        output.verify()
        asm = str(output.mlir_module)


if __name__ == "__main__":
    unittest.main()
