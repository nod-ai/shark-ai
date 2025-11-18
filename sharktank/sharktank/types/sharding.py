# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how a tensor, ops, layers and blocks are
sharded."""

from typing import TYPE_CHECKING, Optional

from abc import ABC, abstractmethod
from amdsharktank.utils import tree
from amdsharktank.types.theta import Theta, flat_to_nested_dict

if TYPE_CHECKING:
    from amdsharktank.layers.configs import LlamaModelConfig


class Sharding(ABC):
    def __init__(self):
        pass


class TensorSharding(Sharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count


class Unsharded(TensorSharding):
    def __init__(self):
        super().__init__(shard_count=1)


class Replicated(TensorSharding):
    def __init__(self, shard_count: int):
        super().__init__(shard_count=shard_count)


class Split(TensorSharding):
    def __init__(self, *, shard_count: int, shard_dim: int):
        super().__init__(shard_count=shard_count)
        self.shard_dim = shard_dim


class Ignore(TensorSharding):
    """When a theta is sharded, a tensor or a branch with this sharding type will be
    ignored.
    It will not appear in the resulting sharded theta.
    This is not strictly a TensorSharding. It will terminate further traversal of a
    branch of a theta tree as well."""

    def __init__(self):
        super().__init__(shard_count=0)


class ThetaSharding(dict):
    """Sharding for each tensor in a theta.
    It is of type dict[str, "ThetaSharding" | TensorSharding].
    """

    def __init__(self, *args, **kwargs):
        d = flat_to_nested_dict(dict(*args, **kwargs))
        for k, v in d.items():
            d[k] = tree.map_nodes(
                tree=v,
                f=lambda x: x
                if isinstance(
                    x,
                    (
                        TensorSharding,
                        ThetaSharding,
                    ),
                )
                else ThetaSharding(x),
            )
        super().__init__(d)


class ThetaLayerSharding(Sharding):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def theta_sharding(self) -> ThetaSharding:
        """Returns the leaf tensor shardings.
        The nested structure would match the one of a corresponding theta for this
        layer.

        ```python
        from amdsharktank.ops import reshard
        theta = ...
        theta_layer_sharding = ...
        theta_sharding = theta_layer_sharding.theta_sharding()
        sharded_theta = reshard(theta, theta_sharding)
        ```
        """
        ...


class AttentionFFNBlockSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int, model_arch: str):
        super().__init__()
        self.shard_count = shard_count
        self.model_arch = model_arch

    def theta_sharding(self) -> ThetaSharding:
        if self.model_arch == "llama":
            result = PagedLlamaAttentionBlockSharding(self.shard_count).theta_sharding()
            result.update(FFNSharding(self.shard_count).theta_sharding())
            result.update(
                {
                    # The size of this is the token embedding length, which is not a memory
                    # space concern if replicated.
                    "ffn_norm": RmsNormReplicatedSharding(
                        self.shard_count
                    ).theta_sharding()
                }
            )
        elif self.model_arch == "deepseek2":
            result = LatentAttentionBlockSharding(self.shard_count).theta_sharding()
            result.update(FFNSharding(self.shard_count).theta_sharding())
            result.update(
                {
                    # The size of this is the token embedding length, which is not a memory
                    # space concern if replicated.
                    "ffn_norm": RmsNormReplicatedSharding(
                        self.shard_count
                    ).theta_sharding()
                }
            )
            result.update(
                MoeBlockSharding(self.shard_count, self.model_arch).theta_sharding()
            )
        elif self.model_arch == "grok":
            result = PagedLlamaAttentionBlockSharding(self.shard_count).theta_sharding()
            result.update(
                {
                    # The size of this is the token embedding length, which is not a memory
                    # space concern if replicated.
                    "ffn_norm": RmsNormReplicatedSharding(
                        self.shard_count
                    ).theta_sharding()
                }
            )
            result.update(
                MoeBlockSharding(self.shard_count, self.model_arch).theta_sharding()
            )
        return result


class Conv2DSplitOutputChannelSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Split(shard_count=self.shard_count, shard_dim=0),
                "bias": Split(shard_count=self.shard_count, shard_dim=0),
            }
        )


class FFNSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int, suffix: str = ""):
        super().__init__()
        self.shard_count = shard_count
        self.suffix = suffix

    def theta_sharding(self) -> ThetaSharding:
        ffn_gate = "ffn_gate" + self.suffix
        ffn_up = "ffn_up" + self.suffix
        ffn_down = "ffn_down" + self.suffix

        return ThetaSharding(
            {
                ffn_gate: LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                ffn_up: LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                ffn_down: LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )


class ExpertParallelRoutedExpertsSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "ffn_gate_exps": LinearSplitBatchWeightAndBiasSharding(
                    shard_count=self.shard_count,
                ).theta_sharding(),
                "ffn_up_exps": LinearSplitBatchWeightAndBiasSharding(
                    shard_count=self.shard_count,
                ).theta_sharding(),
                "ffn_down_exps": LinearSplitBatchWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "exp_probs_b": Ignore(),
            }
        )


class MoeBlockSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int, model_arch: str):
        super().__init__()
        self.shard_count = shard_count
        self.model_arch = model_arch

    def theta_sharding(self) -> ThetaSharding:
        result = ThetaSharding(
            {
                "ffn_gate_inp": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
        if self.model_arch == "deepseek2":
            result.update(
                FFNSharding(self.shard_count, suffix="_shexp").theta_sharding()
            )
        result.update(
            ExpertParallelRoutedExpertsSharding(self.shard_count).theta_sharding()
        )
        result.update(
            {
                "layer_output_norm": RmsNormReplicatedSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )
        return result


class GroupNormSplitChannelSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Split(shard_count=self.shard_count, shard_dim=0),
                "bias": Split(shard_count=self.shard_count, shard_dim=0),
            }
        )


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
                "attn_q_a_norm": RmsNormReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "attn_kv_a_norm": RmsNormReplicatedSharding(
                    self.shard_count
                ).theta_sharding(),
                "attn_q_a": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_q_b": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count,
                ).theta_sharding(),
                "attn_kv_a_mqa": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "attn_kv_b": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count,
                ).theta_sharding(),
                "attn_output": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )


class LinearLayerSharding(ThetaLayerSharding):
    def __init__(
        self, premul_input: TensorSharding, weight: TensorSharding, bias: TensorSharding
    ):
        super().__init__()
        self.premul_input = premul_input
        self.weight = weight
        self.bias = bias

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "premul_input": self.premul_input,
                "weight": self.weight,
                "bias": self.bias,
            }
        )


class LinearReplicatedWeightAndBiasSharding(LinearLayerSharding):
    def __init__(self, shard_count: int, weight_and_bias_spit_dim: int = 0):
        """The linear operation is replicated across devices"""
        super().__init__(
            premul_input=Replicated(shard_count=shard_count),
            weight=Replicated(shard_count=shard_count),
            bias=Replicated(shard_count=shard_count),
        )


class LinearSplitParallelWeightAndBiasSharding(LinearLayerSharding):
    def __init__(self, shard_count: int, weight_and_bias_spit_dim: int = 0):
        """Split one parallel dimension for both the weight and bias.
        Since the weight is transposed before multiplying, the weight parallel
        dimension is the same as the output(bias) dimension."""
        super().__init__(
            premul_input=Replicated(shard_count=shard_count),
            weight=Split(shard_count=shard_count, shard_dim=weight_and_bias_spit_dim),
            bias=Split(shard_count=shard_count, shard_dim=weight_and_bias_spit_dim),
        )


class LinearSplitBatchWeightAndBiasSharding(LinearLayerSharding):
    def __init__(self, shard_count: int, weight_and_bias_spit_dim: int = 0):
        """Split one batch dimension for both the weight and bias.
        Since the weight is transposed before multiplying."""
        super().__init__(
            premul_input=Replicated(shard_count=shard_count),
            weight=Split(shard_count=shard_count, shard_dim=weight_and_bias_spit_dim),
            bias=Split(shard_count=shard_count, shard_dim=weight_and_bias_spit_dim),
        )


class LinearSplitReductionDimSharding(LinearLayerSharding):
    def __init__(self, shard_count: int, reduction_dim: int = 1):
        super().__init__(
            premul_input=Replicated(shard_count=shard_count),
            weight=Split(shard_count=shard_count, shard_dim=reduction_dim),
            bias=Replicated(shard_count=shard_count),
        )


class LlamaSharding(ThetaLayerSharding):
    """Shards the input channel and output channels of the convolutions."""

    def __init__(self, shard_count: int, attention_block_count: int, model_arch: str):
        super().__init__()
        self.shard_count = shard_count
        self.attention_block_count = attention_block_count
        self.model_arch = model_arch

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
                            self.shard_count,
                            model_arch=self.model_arch,
                        ).theta_sharding()
                        for i in range(self.attention_block_count)
                    }
                )
            }
        )
        return result


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
                "kv_cache.quantizer": Replicated(shard_count=self.shard_count),
            }
        )


class RmsNormReplicatedSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Replicated(shard_count=self.shard_count),
            }
        )


class TokenEmbeddingLayerReplicatedSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Replicated(shard_count=self.shard_count),
            }
        )


def shard_theta(
    theta: Theta,
    config: Optional["LlamaModelConfig"] = None,
    sharding: ThetaLayerSharding = None,
) -> Theta:
    assert config or sharding, "shard_theta requires config or sharding"
    if sharding is None:
        sharding = LlamaSharding(
            shard_count=config.tensor_parallelism_size,
            attention_block_count=config.hp.block_count,
            model_arch=config.hp.model_arch,
        )
    from amdsharktank import ops

    return ops.reshard(
        theta,
        spec=sharding,
    )
