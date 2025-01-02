# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import torch
import pytest
from sharktank.models.flux.flux import (
    FluxModelV1,
    FluxParams,
)
from sharktank.models.flux.export import (
    export_flux_transformer_from_hugging_face,
)
from sharktank.models.flux.testing import export_dev_random_single_layer
import sharktank.ops as ops
from sharktank.layers.testing import (
    make_rand_torch,
)
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Dataset, Theta
from sharktank.utils.testing import TempDirTestBase
from sharktank.utils.hf_datasets import get_dataset

logging.basicConfig(level=logging.DEBUG)
with_flux_data = pytest.mark.skipif("not config.getoption('with_flux_data')")


# TODO: Refactor this to a function that generates random toy weights, possibly
# to another file
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


def make_random_theta(dtype: torch.dtype):
    return Theta(
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
                data=make_rand_torch(
                    (patch_size * patch_size * out_channels,), dtype=dtype
                )
            ),
            "final_layer.adaLN_modulation.1.weight": DefaultPrimitiveTensor(  #
                data=make_rand_torch((hidden_size * 2, hidden_size), dtype=dtype)
            ),
            "final_layer.adaLN_modulation.1.bias": DefaultPrimitiveTensor(  #
                data=make_rand_torch((hidden_size * 2,), dtype=dtype)
            ),
        }
    )


class FluxTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24
        self.batch_size = 5

    def testExportDevRandomSingleLayerBf16(self):
        export_dev_random_single_layer(
            dtype=torch.bfloat16,
            batch_sizes=[1],
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_flux_data
    def testExportSchnellTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-schnell/black-forest-labs-transformer",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_flux_data
    def testExportDevTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )


if __name__ == "__main__":
    unittest.main()
