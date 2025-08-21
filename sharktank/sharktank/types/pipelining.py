# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Specifications describing how
"""

from sharktank.types import (
    AnyTensor,
    ReplicatedTensor,
    ShardedTensor,
    Theta,
)


def distribute_blocks_uniformly_over_pipeline_stages(
    block_count: int, pipeline_parallelism_size: int, tensor_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    Default distribution procedure for blocks over pipeline stages.
    This does not take into account any differences in computation time between blocks.
    Nor does it account for any pre- and post-processing (e.g. token embedding and output normalization).

    It is assumed that if block_count % pipeline_parallelism_size != 0, then any stages with more blocks than
    the average will have enough memory to hold the additional blocks.

    Args:
        block_count: The number of blocks to distribute.
        pipeline_parallelism_size: The number of pipeline stages to distribute the blocks over.
        tensor_parallelism_size: The number of devices to distribute each block over.

    Returns:
        A tuple containing:
            - A list mapping each block to its corresponding pipeline stage.
            - A list mapping each pipeline stage to the devices it uses.
            If pipeline_parallelism_size is 1, both lists will be None.

    """

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
    block_data: dict[str, AnyTensor], new_devices: list[int]
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


def pipeline_parallelize_llm_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    In-place pipeline parallelise a theta for an LLM.
    Pipeline parallelize theta for LLM.
    Both DeepSeek and Llama.

    Args:
        theta: The Theta object containing the model.
        pipeline_parallelism_size: The number of pipeline stages to distribute the blocks over.

    Returns:
        A tuple containing:
            - A list mapping each block to its corresponding pipeline stage.
            - A list mapping each pipeline stage to the devices it uses.
            If pipeline_parallelism_size is 1, both lists will be None.
    """
    if pipeline_parallelism_size == 1:
        return None, None

    _t = theta.tensor("token_embd")["weight"]
    tensor_parallelism_size = _t.shard_count if isinstance(_t, ShardedTensor) else 1

    block_indices = [int(bi) for bi in theta.tensor("blk").keys()]
    assert (
        bi == i for i, bi in enumerate(block_indices)
    ), "Blocks assumed to be numbered contiguously from [0, N-1]"

    (
        block_to_pipeline_stage,
        pipeline_stage_to_devices,
    ) = distribute_blocks_uniformly_over_pipeline_stages(
        len(block_indices), pipeline_parallelism_size, tensor_parallelism_size
    )

    for block_index, pipeline_stage in enumerate(block_to_pipeline_stage):
        devices = pipeline_stage_to_devices[pipeline_stage]

        block_data = theta.tensor("blk", block_index)
        for t_name in block_data.keys():
            parallelize_in_place(block_data[t_name], devices)

    parallelize_in_place(theta.tensor("token_embd"), pipeline_stage_to_devices[0])
    parallelize_in_place(theta.tensor("output_norm"), pipeline_stage_to_devices[-1])
    parallelize_in_place(theta.tensor("output"), pipeline_stage_to_devices[-1])

    return block_to_pipeline_stage, pipeline_stage_to_devices
