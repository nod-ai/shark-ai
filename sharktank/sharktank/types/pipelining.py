# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Specifications describing how
"""

from iree.turbine.aot import DeviceTensorTrait, ExternalTensorTrait
from sharktank.types import (
    DefaultPrimitiveTensor,
    PrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    Theta,
)


from typing import Tuple


def pipeline_parallelize_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[tuple[int, ...], ...]:
    """Pipeline parallelize theta for Llama."""
    # TODO: Still modifies the shards, but the signature doesn't imply this
    def parallelize_weight(
        weight: ShardedTensor, new_devices: Tuple[int, ...]
    ) -> ShardedTensor:
        (old_shards, old_devices) = (
            ([weight], (0,))
            if isinstance(weight, PrimitiveTensor)
            else (weight.shards, weight.devices)
        )
        new_shards = ShardedTensor.move_shards_to_new_devices(
            old_shards, old_devices=old_devices, new_devices=new_devices
        )

        for i, (old_shard, new_shard) in enumerate(zip(old_shards, new_shards)):
            DeviceTensorTrait(new_devices[i]).set(new_shard._data)
            if old_tensor_trait := ExternalTensorTrait.get(old_shard._data):
                ExternalTensorTrait(
                    old_tensor_trait.external_scope,
                    old_tensor_trait.external_name,
                ).set(new_shard._data)

        return (
            ReplicatedTensor(ts=new_shards, name=weight.name, devices=new_devices)
            if isinstance(weight, PrimitiveTensor)
            else weight.clone(ts=new_shards, devices=new_devices)
        )

    _t = theta.tensor("token_embd")["weight"]
    shard_count = 1 if isinstance(_t, DefaultPrimitiveTensor) else _t.shard_count
    num_blocks = len(theta.tensor("blk"))

    block_to_device_lookup = []
    block_indices = sorted(theta.tensor("blk").keys(), key=lambda item: int(item))
    assert (
        bi == i for i, bi in enumerate(block_indices)
    ), "Blocks assumed to be numbered contiguously from [0, N-1]"
    for blk_idx in block_indices:
        pp_group = int(int(blk_idx) * pipeline_parallelism_size / num_blocks)
        zero_4_group = shard_count * pp_group
        devices = tuple(i + zero_4_group for i in range(shard_count))
        block_to_device_lookup.append(devices)

        block_data = theta.tensor("blk", blk_idx)
        for t_name in block_data.keys():
            block_data[t_name]["weight"] = parallelize_weight(
                block_data[t_name]["weight"], devices
            )

    theta.tensor("token_embd")["weight"] = parallelize_weight(
        theta.tensor("token_embd")["weight"], block_to_device_lookup[0]
    )
    theta.tensor("output_norm")["weight"] = parallelize_weight(
        theta.tensor("output_norm")["weight"], block_to_device_lookup[-1]
    )
    theta.tensor("output")["weight"] = parallelize_weight(
        theta.tensor("output")["weight"], block_to_device_lookup[-1]
    )

    return tuple(block_to_device_lookup)
