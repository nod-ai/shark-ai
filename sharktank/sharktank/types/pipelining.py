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

from sharktank.types.tensors import QuantizedTensor


def pipeline_parallelize_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    Pipeline parallelize theta for LLM.
    Both DeepSeek and Llama.
    """
    if pipeline_parallelism_size == 1:
        return None, None

    def parallelize_in_place(
        block_data: dict[str, ShardedTensor | PrimitiveTensor | QuantizedTensor],
        new_devices: Tuple[int, ...],
    ) -> None:
        """
        Parallelize the block data in place.
        """
        keys = list(block_data.keys())
        assert len(keys) == 1
        for key in keys:
            tensor = block_data[key]

            # TODO: Handle Quantizer here

            old_shards, old_devices = [tensor], (0,)
            if isinstance(tensor, ShardedTensor):
                old_shards, old_devices = tensor.shards, tensor.devices

            new_shards = ShardedTensor.move_shards_to_new_devices(
                old_shards, old_devices=old_devices, new_devices=new_devices
            )

            # TODO: Need to copy device and external tensor traits of Layout conponents.
            #       Might be best to have each class handle that itself so this doesn't become a long chain of if-statements.
            for i, (old_shard, new_shard) in enumerate(zip(old_shards, new_shards)):
                DeviceTensorTrait(new_devices[i]).set(new_shard._data)
                if old_tensor_trait := ExternalTensorTrait.get(old_shard._data):
                    ExternalTensorTrait(
                        old_tensor_trait.external_scope,
                        old_tensor_trait.external_name,
                    ).set(new_shard._data)

            if isinstance(tensor, ShardedTensor):
                new_tensor = tensor.clone(ts=new_shards, devices=new_devices)
            else:
                new_tensor = ReplicatedTensor(
                    ts=new_shards, name=tensor.name, devices=new_devices
                )

            block_data[key] = new_tensor

    _t = theta.tensor("token_embd")["weight"]
    shard_count = _t.shard_count if isinstance(_t, ShardedTensor) else 1

    block_indices = theta.tensor("blk").keys()
    block_count = len(block_indices)

    block_to_pipeline = [
        i * pipeline_parallelism_size // block_count for i in range(block_count)
    ]
    pipeline_to_devices = [
        [p * shard_count + d for d in range(shard_count)]
        for p in range(pipeline_parallelism_size)
    ]

    assert (
        bi == i for i, bi in enumerate(block_indices)
    ), "Blocks assumed to be numbered contiguously from [0, N-1]"
    for blk_idx in block_indices:
        blk_idx = int(blk_idx)
        pipeline = block_to_pipeline[blk_idx]
        devices = pipeline_to_devices[pipeline]

        # TODO: Make sure this maintains names
        block_data = theta.tensor("blk", blk_idx)
        for t_name in block_data.keys():
            parallelize_in_place(block_data[t_name], devices)

    parallelize_in_place(theta.tensor("token_embd"), pipeline_to_devices[0])
    parallelize_in_place(theta.tensor("output_norm"), pipeline_to_devices[-1])
    parallelize_in_place(theta.tensor("output"), pipeline_to_devices[-1])

    return block_to_pipeline, pipeline_to_devices
