# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Specifications describing how
"""

from iree.turbine.aot import DeviceTensorTrait
from sharktank.types import (
    AnyTensor,
    ReplicatedTensor,
    ShardedTensor,
    Theta,
)

from typing import Tuple


def default_distribute_blocks_for_pipeline_parallelism(
    block_count: int,
    pipeline_parallelism_size: int,
    tensor_parallelism_size: int,
) -> tuple[list[int] | None, list[list[int]] | None]:
    """Assign blocks/layers to pipeline stages and devices(ordinals) to stages."""
    if pipeline_parallelism_size == 1:
        return None, None

    block_to_pipeline_stage = [
        i * pipeline_parallelism_size // block_count for i in range(block_count)
    ]
    pipeline_stage_to_devices = [
        [p * tensor_parallelism_size + d for d in range(tensor_parallelism_size)]
        for p in range(pipeline_parallelism_size)
    ]
    return block_to_pipeline_stage, pipeline_stage_to_devices


def parallelize_in_place(
    block_data: dict[str, AnyTensor],
    new_devices: Tuple[int, ...],
) -> None:
    """
    Parallelize the theta data in place.
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

        for shard, device in zip(new_tensor.shards, new_tensor.devices, strict=True):
            DeviceTensorTrait(device).set(shard.as_torch())

        block_data[block_key] = new_tensor


def pipeline_parallelize_llm_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    Pipeline parallelize theta for LLM.
    Both DeepSeek and Llama.
    """
    if pipeline_parallelism_size == 1:
        return None, None

    _t = theta.tensor("token_embd")["weight"]
    shard_count = _t.shard_count if isinstance(_t, ShardedTensor) else 1

    block_indices = theta.tensor("blk").keys()
    block_count = len(block_indices)

    (
        block_to_pipeline_stage,
        pipeline_stage_to_devices,
    ) = default_distribute_blocks_for_pipeline_parallelism(
        block_count=block_count,
        pipeline_parallelism_size=pipeline_parallelism_size,
        tensor_parallelism_size=shard_count,
    )

    assert (
        bi == i for i, bi in enumerate(block_indices)
    ), "Blocks assumed to be numbered contiguously from [0, N-1]"
    for blk_idx in block_indices:
        blk_idx = int(blk_idx)
        stage = block_to_pipeline_stage[blk_idx]
        devices = pipeline_stage_to_devices[stage]

        block_data = theta.tensor("blk", blk_idx)
        for t_name in block_data.keys():
            parallelize_in_place(block_data[t_name], devices)

    parallelize_in_place(theta.tensor("token_embd"), pipeline_stage_to_devices[0])
    parallelize_in_place(theta.tensor("output_norm"), pipeline_stage_to_devices[-1])
    parallelize_in_place(theta.tensor("output"), pipeline_stage_to_devices[-1])

    return block_to_pipeline_stage, pipeline_stage_to_devices
