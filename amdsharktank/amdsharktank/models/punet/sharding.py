# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how blocks/layers of punet are sharded."""

from amdsharktank.types.sharding import *


class ResnetBlock2DSplitOutputChannelsSharding(ThetaLayerSharding):
    """Shards the input channel and output channels of the convolutions."""

    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        result = ThetaSharding(
            {
                "norm1": GroupNormSplitChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv1": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "norm2": GroupNormSplitChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv2": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "time_emb_proj": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "conv_shortcut": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
        return result


class UpDownSample2DSplitChannelSharding(ThetaLayerSharding):
    """Splits the output channels dimension of the convolution.

    Sharding spec for `.layers.Upsample2D` or `.layers.Downsample2D`.
    """

    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        result = ThetaSharding(
            {
                "conv": Conv2DSplitOutputChannelSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
        return result


class UpDownBlock2DSplitChannelsSharing(ThetaLayerSharding):
    """Splits the output channels dimension of the convolution.

    Sharding spec for `.layers.Downsample2D` or `.layers.Upsample2D`.
    """

    def __init__(
        self,
        shard_count: int,
        resnet_layers_count: int,
        upsamplers_count: int = 0,
        downsamplers_count: int = 0,
    ):
        super().__init__()
        self.shard_count = shard_count
        self.resnet_layers_count = resnet_layers_count
        self.upsamplers_count = upsamplers_count
        self.downsamplers_count = downsamplers_count

    def theta_sharding(self) -> ThetaSharding:
        d = {}
        for i in range(self.upsamplers_count):
            d[f"upsamplers.{i}"] = UpDownSample2DSplitChannelSharding(
                shard_count=self.shard_count
            ).theta_sharding()
        for i in range(self.downsamplers_count):
            d[f"downsamplers.{i}"] = UpDownSample2DSplitChannelSharding(
                shard_count=self.shard_count
            ).theta_sharding()
        for i in range(self.resnet_layers_count):
            d[f"resnets.{i}"] = ResnetBlock2DSplitOutputChannelsSharding(
                shard_count=self.shard_count
            ).theta_sharding()
        return ThetaSharding(d)
