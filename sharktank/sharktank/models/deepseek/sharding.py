# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how the Llama model is sharded."""

from ...types.sharding import *
from ...types import Theta
from ... import ops


class LatentAttentionBlockSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                # The size of this is the token embedding length, which is not a memory
                # space concern if replicated even for all attention blocks.
                "attn_norm": RmsNormReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "q_norm": RmsNormReplicatedSharding(self.shard_count).theta_sharding(),
                "kv_norm": RmsNormReplicatedSharding(self.shard_count).theta_sharding(),
                "wq": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "wq_a": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "wq_b": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "wkv_a": LinearReplicatedWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "wkv_b": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "wo": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_output": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )


def shard_theta(theta: Theta, sharding: ThetaLayerSharding) -> Theta:
    return ops.reshard(theta, sharding)
