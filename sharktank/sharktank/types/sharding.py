# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how a tensor, ops, layers and blocks are
sharded."""

from abc import ABC, abstractmethod
from sharktank.utils import tree
from sharktank.types.theta import flat_to_nested_dict


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
        from sharktank.ops import reshard
        theta = ...
        theta_layer_sharding = ...
        theta_sharding = theta_layer_sharding.theta_sharding()
        sharded_theta = reshard(theta, theta_sharding)
        ```
        """
        ...


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
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "ffn_gate": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "ffn_up": LinearSplitParallelWeightAndBiasSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
                "ffn_down": LinearSplitReductionDimSharding(
                    shard_count=self.shard_count
                ).theta_sharding(),
            }
        )


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


class LinearSplitReductionDimSharding(LinearLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__(
            premul_input=Replicated(shard_count=shard_count),
            weight=Split(shard_count=shard_count, shard_dim=1),
            bias=Replicated(shard_count=shard_count),
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
