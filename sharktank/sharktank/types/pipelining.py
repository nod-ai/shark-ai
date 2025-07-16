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
    PrimitiveTensor,
    QuantizedTensor,
    ReplicatedTensor,
    ShardedTensor,
    Theta,
)


from typing import Tuple


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
        for block_key in list(block_data.keys()):
            tensor = block_data[block_key]
            shards = tensor.shards if isinstance(tensor, ShardedTensor) else [tensor]

            if isinstance(tensor, ShardedTensor):
                new_tensor = tensor.clone(ts=shards, devices=new_devices)
            else:
                new_tensor = ReplicatedTensor(
                    ts=shards, name=tensor.name, devices=new_devices
                )

            block_data[block_key] = new_tensor

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

        block_data = theta.tensor("blk", blk_idx)
        for t_name in block_data.keys():
            parallelize_in_place(block_data[t_name], devices)

    parallelize_in_place(theta.tensor("token_embd"), pipeline_to_devices[0])
    parallelize_in_place(theta.tensor("output_norm"), pipeline_to_devices[-1])
    parallelize_in_place(theta.tensor("output"), pipeline_to_devices[-1])

    return block_to_pipeline, pipeline_to_devices
