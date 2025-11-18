# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import List
from parameterized import parameterized

import torch

from amdsharktank.models.punet.testing import make_up_down_block_2d_theta
from amdsharktank.models.punet.layers import UpDownBlock2D
from amdsharktank.models.punet.sharding import UpDownBlock2DSplitChannelsSharing
from amdsharktank.types import *
from amdsharktank import ops
from amdsharktank.types.tensors import flatten_tensor_tree
from amdsharktank.utils.testing import assert_tensor_close


class UpBlock2DTest(unittest.TestCase):
    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def testUpDownBlock2DSplitInputAndOutputChannelsSharding(self, is_up_block: bool):
        with torch.random.fork_rng():
            torch.random.manual_seed(12345)
            torch.set_default_dtype(torch.float32)
            batches = 2
            channels = 12
            height = 11
            width = 13
            kernel_height = 5
            kernel_width = 5
            input_time_emb_shape = [batches, channels]
            norm_groups = 2
            eps = 0.01
            shard_count = 2
            resnet_layers = 2
            theta = make_up_down_block_2d_theta(
                channels=channels,
                kernel_height=kernel_height,
                kernel_width=kernel_width,
                input_time_emb_channels=channels,
                resnet_layers=resnet_layers,
                is_up_block=is_up_block,
            )
            block = UpDownBlock2D(
                theta=theta,
                num_layers=resnet_layers,
                resnet_eps=eps,
                resnet_act_fn="relu",
                resnet_groups=norm_groups,
                resnet_out_scale_factor=None,
                resnet_time_scale_shift="default",
                temb_channels=channels,
                dropout=0.0,
                add_upsample=is_up_block,
                add_downsample=not is_up_block,
                downsample_padding=1,
            )
            input_image = torch.rand(
                batches,
                channels // 2,
                height,
                width,
            )
            input_time_emb = torch.rand(input_time_emb_shape)
            res_hidden_states_tuple = [
                torch.rand(
                    batches,
                    channels // 2,
                    height - 2 * i * (kernel_height // 2),
                    width - 2 * i * (kernel_width // 2),
                )
                for i in range(resnet_layers)
            ]
            res_hidden_states_tuple.reverse()

            expected_result = flatten_tensor_tree(
                block(
                    hidden_states=input_image,
                    temb=input_time_emb,
                    res_hidden_states_tuple=res_hidden_states_tuple,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            )

            sharding_spec = UpDownBlock2DSplitChannelsSharing(
                shard_count=shard_count,
                resnet_layers_count=resnet_layers,
                upsamplers_count=1 if is_up_block else 0,
                downsamplers_count=1 if not is_up_block else 0,
            )
            sharded_theta = ops.reshard(theta, sharding_spec)
            sharded_block = UpDownBlock2D(
                theta=sharded_theta,
                num_layers=resnet_layers,
                resnet_eps=eps,
                resnet_act_fn="relu",
                resnet_groups=norm_groups,
                resnet_out_scale_factor=None,
                resnet_time_scale_shift="default",
                temb_channels=channels,
                dropout=0.0,
                add_upsample=is_up_block,
                add_downsample=not is_up_block,
                downsample_padding=1,
            )
            sharded_input_image = ops.reshard_split(
                input_image, dim=1, count=shard_count
            )
            sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)
            sharded_res_hidden_states_tuple = [
                ops.reshard_split(tensor, dim=1, count=shard_count)
                for tensor in res_hidden_states_tuple
            ]
            sharded_result = flatten_tensor_tree(
                sharded_block(
                    hidden_states=sharded_input_image,
                    temb=sharded_input_time_emb,
                    res_hidden_states_tuple=sharded_res_hidden_states_tuple,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            )
            assert all(
                [
                    isinstance(r, SplitPrimitiveTensor)
                    and r.shard_dim == 1
                    and r.shard_count == shard_count
                    for r in sharded_result
                ]
            )
            actual_result = [ops.unshard(r) for r in sharded_result]
            assert len(expected_result) == len(actual_result)
            for actual, expected in zip(actual_result, expected_result):
                assert_tensor_close(actual, expected, rtol=0, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
