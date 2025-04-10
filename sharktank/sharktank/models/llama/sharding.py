# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how the Llama model is sharded."""

from sharktank.layers.configs import LlamaModelConfig
from ...types.sharding import *
from ...types import Theta
from ... import ops


class PagedLlamaAttentionBlockSharding(ThetaLayerSharding):
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
                "attn_q": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_k": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_v": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_output": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )


class AttentionFFNBlockSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        result = PagedLlamaAttentionBlockSharding(self.shard_count).theta_sharding()
        result.update(FFNSharding(self.shard_count).theta_sharding())
        result.update(
            {
                # The size of this is the token embedding length, which is not a memory
                # space concern if replicated.
                "ffn_norm": RmsNormReplicatedSharding(self.shard_count).theta_sharding()
            }
        )
        return result


class LlamaSharding(ThetaLayerSharding):
    """Shards the input channel and output channels of the convolutions."""

    def __init__(self, shard_count: int, attention_block_count: int):
        super().__init__()
        self.shard_count = shard_count
        self.attention_block_count = attention_block_count

    def theta_sharding(self) -> ThetaSharding:
        result = ThetaSharding(
            {
                # Replicate the vocabulary. For llama 1-3 this will require 0.5 GiB.
                # For devices with large memory this may be an acceptable tradeoff where
                # we save on communication by not all-gathering the result afterwards.
                # The computation is just indexing and replication is not a concern.
                # Alternatively, we can try splitting the index dimension,
                # this would require custom logic for indexing partitioning and gathering.
                "token_embd": TokenEmbeddingLayerReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "rope_freqs": Ignore(),
                "output_norm": RmsNormReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "output": LinearSplitReductionDimSharding(
                    self.shard_count
                ).theta_sharding(),
            }
        )
        result.update(
            {
                "blk": ThetaSharding(
                    {
                        f"{i}": AttentionFFNBlockSharding(
                            self.shard_count
                        ).theta_sharding()
                        for i in range(self.attention_block_count)
                    }
                )
            }
        )
        return result


def shard_theta(theta: Theta, config: LlamaModelConfig) -> Theta:
    return ops.reshard(
        theta,
        LlamaSharding(
            shard_count=config.tensor_parallelism_size,
            attention_block_count=config.hp.block_count,
        ),
    )
