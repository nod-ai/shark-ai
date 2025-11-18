# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import List

import torch

from amdsharktank.models.punet.testing import make_resnet_block_2d_theta
from amdsharktank.models.punet.layers import ResnetBlock2D
from amdsharktank.types import *
from amdsharktank.models.punet.sharding import ResnetBlock2DSplitOutputChannelsSharding
from amdsharktank import ops
from amdsharktank.utils.testing import assert_tensor_close


class ResnetBlockTest(unittest.TestCase):
    def testResnetBlock2DSplitInputAndOutputChannelsSharding(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(12345)
            torch.set_default_dtype(torch.float32)
            batches = 2
            in_channels = 6
            out_channels = [12, 8]
            height = 11
            width = 13
            kernel_height = 5
            kernel_width = 5
            input_time_emb_shape = [batches, 8]
            norm_groups = 2
            eps = 0.01
            shard_count = 2
            theta = make_resnet_block_2d_theta(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                input_time_emb_channels=input_time_emb_shape[1],
            )
            resnet_block = ResnetBlock2D(
                theta=theta,
                groups=norm_groups,
                eps=eps,
                non_linearity="relu",
                output_scale_factor=None,
                dropout=0.0,
                temb_channels=input_time_emb_shape[1],
                time_embedding_norm="default",
            )
            input_image = torch.rand(
                batches,
                in_channels,
                height,
                width,
            )
            input_time_emb = torch.rand(input_time_emb_shape)
            expected_result = resnet_block(input_image, input_time_emb)

            sharding_spec = ResnetBlock2DSplitOutputChannelsSharding(
                shard_count=shard_count
            )
            sharded_theta = ops.reshard(theta, sharding_spec)
            sharded_resnet_block = ResnetBlock2D(
                theta=sharded_theta,
                groups=norm_groups,
                eps=eps,
                non_linearity="relu",
                output_scale_factor=None,
                dropout=0.0,
                temb_channels=input_time_emb_shape[1],
                time_embedding_norm="default",
            )
            sharded_input_image = ops.reshard_split(
                input_image, dim=1, count=shard_count
            )
            sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)
            sharded_result = sharded_resnet_block(
                sharded_input_image, sharded_input_time_emb
            )
            assert isinstance(sharded_result, SplitPrimitiveTensor)
            assert (
                sharded_result.shard_dim == 1
                and sharded_result.shard_count == shard_count
            )
            actual_result = ops.unshard(sharded_result)
            assert_tensor_close(expected_result, actual_result, rtol=0, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
