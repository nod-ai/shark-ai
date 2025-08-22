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


def transfer_between_blocks_if_needed(
    x: AnyTensor, curr_block: int, theta: Theta
) -> ShardedTensor:
    """
    Function to run between blocks in a model to insert transfer required by pipeline parallelism.

    If transfers are not needed, the input tensor is returned unchanged.

    Args:
        x: The input tensor to process.
        curr_block: The index of the current block.
        theta: The theta object used

    Returns:
        The input tensor, possibly moved to different devices.
    """
    # Unsharded tensors do not need to be moved.
    if not isinstance(x, ShardedTensor):
        return x

    # Last block, nothing to do.
    block_count = len(theta.tensor("blk"))
    if curr_block == block_count - 1:
        return x

    curr_devices = x.devices
    # Grab first weight from next block, all weights in a block are on the same devices.
    next_weights = list(theta.tensor("blk", curr_block + 1).values())[0]["weight"]
    next_devices = next_weights.devices

    # If the current and next devices are the same, nothing to do.
    if all(d_curr == d_next for d_curr, d_next in zip(curr_devices, next_devices)):
        return x

    # Devices are different, need to move shards.
    shards = ShardedTensor.move_shards_to_new_devices(
        x.shards, old_devices=curr_devices, new_devices=next_devices
    )
    return x.clone(ts=shards, devices=next_devices)


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
    Unsharded weights are converted to a ReplicatedTensor with the new devices.
    Weights that are already sharded will be cloned with new devices.
    NOTE: No transfers are performed, only the tagged devices are changed.
          This function should not be traced, and should only be used for setting up the
          Theta after loading from file but before tracing.

    Args:
        block_data: A dictionary containing the block data, where keys are tensor names and values are
                    the corresponding tensors.
        new_devices: A list of devices on which all of the tensors should be placed.

    Returns:
        None: Tensors are modified in-place.
    """
    for block_key in list(block_data.keys()):
        tensor = block_data[block_key]
        if isinstance(tensor, ShardedTensor):
            block_data[block_key] = tensor.clone(devices=new_devices)
        else:
            block_data[block_key] = ReplicatedTensor(
                ts=[tensor], name=tensor.name, devices=new_devices
            )


def pipeline_parallelize_llm_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    In-place pipeline parallelise a theta for an LLM.

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
