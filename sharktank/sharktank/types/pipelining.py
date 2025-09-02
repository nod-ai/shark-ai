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


def get_devices_from_block_tensors(
    curr_block_tensors: dict[str, dict[str, AnyTensor] | AnyTensor]
) -> list[int] | None:
    """
    Helper function to extract devices from the current block tensors.
    Ensures that all tensors in the block are on the same devices, raises an error if they're not.

    Args:
        curr_block_tensors: The tensors associated with the current block. From theta.tensor(...)

    Returns:
        A list of devices if the block is sharded, None if the block is not sharded.
    """

    def iter_equal(A, B):
        return all(a == b for a, b in zip(A, B))

    devices = -1
    for subvals in curr_block_tensors.values():
        tensors = subvals.values() if isinstance(subvals, dict) else [subvals]
        for tensor in tensors:
            if isinstance(tensor, ShardedTensor):
                if devices == -1:
                    devices = tensor.devices
                assert (
                    devices is not None
                ), "Block contains a mix of sharded and unsharded tensors."
                assert iter_equal(devices, tensor.devices), (
                    f"All tensors in a block must be on the same devices."
                    f"Found {devices} and {tensor.devices}."
                )
            else:
                if devices == -1:
                    devices = None
                # TODO: This will fail with QuantizerTensors and PP/TP.
                assert (
                    devices is None
                ), "Block contains a mix of sharded and unsharded tensors."

    assert devices != -1, "Block contains no tensors."
    return devices


def transfer_between_blocks(
    *xs: AnyTensor | None, curr_block_tensors: dict[str, dict[str, AnyTensor]]
) -> ShardedTensor | None | list[ShardedTensor | None]:
    """
    Function to run between blocks in a model to insert transfer required by pipeline parallelism.

    If transfers are not needed, the input tensor is returned unchanged.

    Args:
        x: The input tensor to process.
        curr_block: The tensors associated with the current block.

    Returns:
        The input tensor, possibly moved to different devices.
    """
    new_devices = get_devices_from_block_tensors(curr_block_tensors)

    # Weights are not ShardedTensors, therefor model is not pipelined.
    if new_devices is None:
        return xs

    new_xs = []
    for x in xs:
        if x is None:
            new_xs.append(None)
            continue

        if isinstance(x, ShardedTensor):
            shards = ShardedTensor.move_shards_to_new_devices(
                x.shards, new_devices=new_devices
            )
            new_x = x.clone(ts=shards, devices=new_devices)
        else:
            shards = ShardedTensor.move_shards_to_new_devices(
                (x,), new_devices=new_devices
            )
            new_x = ReplicatedTensor(ts=shards, devices=new_devices)
        new_xs.append(new_x)

    if len(new_xs) == 1:
        return new_xs[0]
    return new_xs


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
