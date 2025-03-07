# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Light-weight encapsulations for various forms of attention KV-caches.

These are not complete abstractions: they are primarily focused on making
tightly coupled transformer blocks a bit less "stringy" with loose tensors
and dims floating around everywhere.
"""

from typing import Optional, Union, List

import abc
import math

import torch

from ..utils.debugging import trace_tensor
from ..types import SplitPrimitiveTensor, ReplicatedTensor
from .. import ops

__all__ = ["PagedKVCache"]


class PagedKVCache:
    """Implementation of a KV cache on top of a 'page table'.

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * block sequence stride (number of sequence positions per block)
    * attention heads
    * attention dimensionality

    Note that the internal page structure matches the organization of the
    model, allowing contiguous individual local reads and writes at a sub-block
    granularity if indexing deeply into the structure.

    When `shard_count > 1`, it would split the `attn_head_count` dimension.
    The page slab is a 1D sharded split tensor.
    It is reinterpreted as a 6D tensor, by working around the lack of sharded
    block-cyclic sharded tensor type.
    """

    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        shard_count: int = 1,
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count
        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        # Some derived values based on attributes.
        self.sub_page_dims = [
            self.transformer_block_count,
            self.cache_partition_count,
            self.block_seq_stride,
            self.attn_head_count // self.shard_count,
            self.attn_head_dim,
        ]
        self.page_slab_flat_dim = math.prod(self.sub_page_dims)
        self.device = device
        self.dtype = dtype

    def unflatten_page_table(
        self, state: list[Union[torch.Tensor, SplitPrimitiveTensor]]
    ) -> Union[torch.Tensor, SplitPrimitiveTensor]:
        """Unflattens the 2D page table to a 6D tensor."""
        assert len(state) == 1, f"Expected 1-element state. Got: {len(state)}"
        page_slab = state[0]
        if self.shard_count == 1:
            assert not isinstance(page_slab, SplitPrimitiveTensor)
            return page_slab.unflatten(1, self.sub_page_dims)
        else:
            assert self.shard_count == page_slab.shard_count
            shards = [
                shard.unflatten(1, self.sub_page_dims) for shard in page_slab.shards
            ]
            return SplitPrimitiveTensor(ts=shards, shard_dim=4)

    def shard_state(
        self, state: List[torch.Tensor]
    ) -> List[Union[torch.Tensor, SplitPrimitiveTensor]]:
        """Shard an unsharded state.
        We can't just split the slab on the sub page dims.
        First it needs to be reinterpreted into the actual shape.
        The split the head dimension, then flatten each shard.
        This is a work-around for the lack of block-cyclic sharded tensor type."""
        if self.shard_count == 1:
            return state

        page_table = state[0].reshape(
            [
                -1,
                self.transformer_block_count,
                self.cache_partition_count,
                self.block_seq_stride,
                self.attn_head_count,
                self.attn_head_dim,
            ]
        )
        sharded_page_table = ops.reshard_split(
            page_table, dim=4, count=self.shard_count
        )
        shards = [
            ops.flatten(shard, start_dim=1) for shard in sharded_page_table.shards
        ]
        flat_sharded_page_table = SplitPrimitiveTensor(ts=shards, shard_dim=1)
        return [flat_sharded_page_table]

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(
        self, page_count: int
    ) -> list[Union[torch.Tensor, SplitPrimitiveTensor]]:
        """Allocates tensor state for a page table for the given capacity in
        pages.
        """
        shards = [
            torch.empty(
                [page_count, self.page_slab_flat_dim],
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.shard_count)
        ]

        if self.shard_count == 1:
            return shards

        return [SplitPrimitiveTensor(ts=shards, shard_dim=1)]

    def read(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        seq_len: int,
        page_ids: Optional[Union[torch.Tensor, ReplicatedTensor]] = None,
    ):
        """Reads K/V caches the page table for the given page_ids.

        Args:
        state: State struct as returned from allocate().
        transformer_block_index: The index of the transformer block accessing
            the cache.
        page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids
            to access.

        Returns the K/V cache partitions, linearized. Note that this reference
        approach to reading by materializing linearly may not be terribly
        efficient unless if the compiler can fuse the gather.
        """
        page_table = self.unflatten_page_table(state)  # 6D

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.cache_partition_count,
            self.block_seq_stride,
            self.attn_head_count // self.shard_count,
            self.attn_head_dim,
        ]

        # Gather both partitions and split post gather. This is more
        # computationally efficient without gather fusion:
        subblock_table = page_table.flatten(start_dim=0, end_dim=1)
        page_stride = self.transformer_block_count

        # Create a tensor filled with the transformer block index
        transformer_block_index_tensor = torch.full(
            (bs, block_seq_len), transformer_block_index
        )

        # Calculate subblock IDs - these need to be in bounds
        max_valid_index = subblock_table.shape[0] - 1

        # Create the indices - multiply page ID by stride and add transformer block index
        subblock_ids = page_ids * page_stride + transformer_block_index_tensor

        # Flatten the indices
        flat_ids = subblock_ids.flatten(0, 1)

        # Ensure all indices are in bounds
        valid_mask = flat_ids <= max_valid_index
        if not torch.all(valid_mask):
            # If any indices are out of bounds, clamp them
            flat_ids = torch.clamp(flat_ids, max=max_valid_index)

        # Now select the entries from the page table
        selected = ops.index_select(subblock_table, 0, flat_ids)

        selected = selected.unflatten(0, blocked_shape[:2])
        key = selected[:, :, 0, :seq_len].flatten(1, 2)[:, :seq_len]
        value = selected[:, :, 1, :seq_len].flatten(1, 2)[:, :seq_len]

        return key, value

    def write_timestep(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        # List of [bs, 1, attn_head_count, attn_head_dim]
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, max_seqlen // block_pos_stride]
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a single batched timestep across all cache partitions.

        Note that this internally loops over the batch size, which cannot be
        dynamic.
        """
        device = self.device
        page_table = self.unflatten_page_table(state)  # 6D
        page_table = page_table.flatten(0, 3)
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count

        # [bs, 1, atten_head_count, attn_head_dim]
        for idx, cache_partition in enumerate(cache_partitions):
            # [bs, 1]
            page_index = seq_positions // self.block_seq_stride

            page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
            page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

            # [1, 1]
            if isinstance(seq_positions, ReplicatedTensor):
                partitions = [
                    torch.tensor(idx).unsqueeze(0)
                    for _ in range(seq_positions.shard_count)
                ]

                transformer_block = [
                    torch.full((bs, 1), transformer_block_index, device=device)
                    for _ in range(seq_positions.shard_count)
                ]

                partitions = ReplicatedTensor(ts=partitions)
                transformer_block = ReplicatedTensor(ts=transformer_block)
            else:
                partitions = torch.tensor(idx).unsqueeze(0)
                transformer_block = torch.full(
                    (bs, 1), transformer_block_index, device=device
                )

            partitions = partitions.repeat(bs, 1)

            index = page_id
            index = index * self.transformer_block_count + transformer_block
            index = index * self.cache_partition_count + partitions
            index = index * self.block_seq_stride + page_offset
            values = ops.to(cache_partition, dtype=page_table.dtype)
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=(index,), values=values_int8)
            else:
                page_table.index_put_(indices=(index,), values=values)

        return

    def write(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        page_table = self.unflatten_page_table(state)  # 6D

        bs, block_seq_len, *_ = page_ids.shape

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        page_stride = self.transformer_block_count * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        for index, partition in enumerate(cache_partitions):
            # Handle case where sequence length doesn't evenly divide by block_seq_stride
            seq_len = partition.shape[1]
            # Calculate how many blocks we need based on the actual sequence length
            full_blocks = seq_len // self.block_seq_stride
            remainder = seq_len % self.block_seq_stride

            # Process full blocks first if any
            if full_blocks > 0:
                # Extract the portion that fits evenly into blocks
                full_block_data = partition[:, : full_blocks * self.block_seq_stride]
                part_block_view = full_block_data.unflatten(
                    1, (full_blocks, self.block_seq_stride)
                )
                part_block_view = part_block_view.flatten(0, 1)

                # Get the corresponding block IDs
                if full_blocks < block_seq_len:
                    # Only use the blocks we need
                    block_subids = base_subblock_ids[:, :full_blocks]
                else:
                    # Use all available blocks, but limit to block_seq_len
                    # This prevents out-of-bounds access when block_seq_len < full_blocks
                    block_subids = base_subblock_ids[
                        :, : min(full_blocks, block_seq_len)
                    ]

                subblock_ids = (
                    (block_subids + index) if index > 0 else block_subids
                ).flatten(0, 1)

                part_block = ops.to(part_block_view, dtype=subblock_table.dtype)

                # Make sure indices are in bounds
                max_valid_index = subblock_table.shape[0] - 1
                valid_mask = subblock_ids <= max_valid_index

                if not torch.all(valid_mask):
                    # If any indices are out of bounds, skip them
                    valid_indices = subblock_ids[valid_mask]
                    valid_part_block = part_block[valid_mask]

                    if (
                        valid_indices.numel() > 0
                    ):  # Only proceed if we have valid indices
                        if subblock_table.dtype == torch.float8_e4m3fnuz:
                            # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                            subblock_table_as_int8 = subblock_table.view(
                                dtype=torch.int8
                            )
                            valid_part_block_int8 = valid_part_block.view(
                                dtype=torch.int8
                            )
                            subblock_table_as_int8.index_copy_(
                                0, valid_indices, valid_part_block_int8
                            )
                        else:
                            subblock_table.index_copy_(
                                0, valid_indices, valid_part_block
                            )
                else:
                    # All indices are valid
                    if subblock_table.dtype == torch.float8_e4m3fnuz:
                        # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                        subblock_table_as_int8 = subblock_table.view(dtype=torch.int8)
                        part_block_as_int8 = part_block.view(dtype=torch.int8)
                        subblock_table_as_int8.index_copy_(
                            0, subblock_ids, part_block_as_int8
                        )
                    else:
                        subblock_table.index_copy_(0, subblock_ids, part_block)

            # Handle remainder if any
            if remainder > 0 and full_blocks < block_seq_len:
                # Extract the remainder
                remainder_data = partition[:, full_blocks * self.block_seq_stride :]
                # Pad to full block size
                padded_remainder = torch.nn.functional.pad(
                    remainder_data, (0, 0, 0, 0, 0, self.block_seq_stride - remainder)
                )

                # Get the block ID for the remainder
                if full_blocks < base_subblock_ids.shape[1]:
                    remainder_block_id = base_subblock_ids[:, full_blocks]
                    if index > 0:
                        remainder_block_id = remainder_block_id + index

                    remainder_block_id = remainder_block_id.flatten()

                    # Check that the remainder_block_id is in bounds
                    if remainder_block_id.item() < len(subblock_table):
                        # Convert and write only the valid part of the remainder
                        padded_remainder = ops.to(
                            padded_remainder.flatten(), dtype=subblock_table.dtype
                        )

                        if subblock_table.dtype == torch.float8_e4m3fnuz:
                            subblock_table_as_int8 = subblock_table.view(
                                dtype=torch.int8
                            )
                            padded_remainder_int8 = padded_remainder.view(
                                dtype=torch.int8
                            )

                            # Get current block data
                            current_block = subblock_table_as_int8[remainder_block_id]

                            # Create mask for the part we want to update (first 'remainder' positions)
                            remainder_indices = torch.arange(
                                current_block.shape[0], device=current_block.device
                            )
                            valid_indices = (
                                remainder_indices
                                < remainder * partition.shape[2] * partition.shape[3]
                            )

                            # Update only the valid part
                            current_block[valid_indices] = padded_remainder_int8[
                                valid_indices
                            ]
                            subblock_table_as_int8[remainder_block_id] = current_block
                        else:
                            # Get current block data
                            current_block = subblock_table[remainder_block_id]

                            # Create mask for the part we want to update (first 'remainder' positions)
                            remainder_indices = torch.arange(
                                current_block.shape[0], device=current_block.device
                            )
                            valid_indices = (
                                remainder_indices
                                < remainder * partition.shape[2] * partition.shape[3]
                            )

                            # Update only the valid part
                            current_block[valid_indices] = padded_remainder[
                                valid_indices
                            ]
                            subblock_table[remainder_block_id] = current_block

    def write_selective(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
        start_positions: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions selectively starting from start_positions.

        This method uses a hybrid approach:
        1. For blocks that are fully after start_positions, uses standard write()
        2. For blocks that are partially after start_positions, selectively updates
        3. For blocks that are fully before start_positions, skips them entirely

        Args:
            state: State struct as returned from allocate().
            cache_partitions: List of tensors to write for each partition.
            transformer_block_index: The transformer block index.
            page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids.
            start_positions: Tensor of [bs] indicating the position to start updating from.
        """
        page_table = self.unflatten_page_table(state)  # 6D
        bs, block_seq_len, *_ = page_ids.shape

        # Determine which blocks need to be updated for each batch element
        # Calculate the start block for each sequence in the batch
        start_blocks = start_positions // self.block_seq_stride

        # Determine positions within start blocks
        start_offsets = start_positions % self.block_seq_stride

        # Setup for indexing into the page table
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        page_stride = self.transformer_block_count * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count

        # Process each partition (K and V)
        for idx, partition in enumerate(cache_partitions):
            # Handle case where sequence length doesn't evenly divide by block_seq_stride
            seq_len = partition.shape[1]
            # Calculate how many blocks we need based on the actual sequence length
            full_blocks = seq_len // self.block_seq_stride
            remainder = seq_len % self.block_seq_stride

            # Reshape partition based on full blocks
            if full_blocks > 0:
                # Extract the portion that fits evenly into blocks
                full_block_data = partition[:, : full_blocks * self.block_seq_stride]
                part_blocked = full_block_data.unflatten(
                    1, (full_blocks, self.block_seq_stride)
                )
            else:
                # Handle the case with no full blocks
                # Create an empty tensor with the right shape
                part_blocked = torch.zeros(
                    (partition.shape[0], 0, self.block_seq_stride)
                    + partition.shape[2:],
                    dtype=partition.dtype,
                    device=partition.device,
                )

            # Process each batch element separately
            for b in range(bs):
                start_block = start_blocks[b].item()
                start_offset = start_offsets[b].item()

                # Skip if there's nothing to update
                if start_block >= full_blocks and (
                    start_block > full_blocks or remainder == 0
                ):
                    continue

                # Process all blocks for this batch element
                for blk in range(min(full_blocks, block_seq_len)):
                    # When extending from position 16 (a block boundary), we
                    # need to make sure all blocks from the start block onwards are updated
                    # We keep blocks before start_block untouched
                    if blk < start_block:
                        # For blocks before the start position, we don't update anything
                        # This preserves the KV cache for tokens before start_positions
                        continue

                    # Get the current block data
                    curr_block = part_blocked[b, blk]
                    page_id = page_ids[b, blk]

                    # For start block, selectively update from start offset
                    if blk == start_block and start_offset > 0:
                        # Calculate index for the page table
                        block_index = (
                            page_id * page_stride
                            + transformer_block_index * transformer_block_stride
                            + idx
                        )

                        # Check if block_index is valid before indexing
                        if block_index < len(subblock_table):
                            # Get current values to preserve before start_offset
                            curr_values = subblock_table[block_index]
                            curr_values = curr_values.reshape(
                                self.block_seq_stride, -1, self.attn_head_dim
                            )

                            # Create updated values - keep original for positions before start_offset
                            updated_values = curr_values.clone()
                            updated_values[start_offset:] = curr_block[start_offset:]

                            # Write back to table
                            updated_flat = updated_values.reshape(-1)
                            if subblock_table.dtype == torch.float8_e4m3fnuz:
                                subblock_table_as_int8 = subblock_table.view(
                                    dtype=torch.int8
                                )
                                updated_flat_int8 = updated_flat.view(dtype=torch.int8)
                                subblock_table_as_int8[block_index] = updated_flat_int8
                            else:
                                subblock_table[block_index] = updated_flat
                        else:
                            # Log or handle the case where the index is out of bounds
                            pass
                    else:
                        # For blocks after start_block, update entire block
                        block_index = (
                            page_id * page_stride
                            + transformer_block_index * transformer_block_stride
                            + idx
                        )
                        # Check if block_index is valid before indexing
                        if block_index < len(subblock_table):
                            block_data = ops.to(
                                curr_block.flatten(), dtype=subblock_table.dtype
                            )

                            if subblock_table.dtype == torch.float8_e4m3fnuz:
                                subblock_table_as_int8 = subblock_table.view(
                                    dtype=torch.int8
                                )
                                block_data_int8 = block_data.view(dtype=torch.int8)
                                subblock_table_as_int8[block_index] = block_data_int8
                            else:
                                subblock_table[block_index] = block_data
                        else:
                            # Log or handle the case where the index is out of bounds
                            # This could happen if the page_ids aren't allocated properly
                            pass

                # Handle remainder block if any
                if remainder > 0 and full_blocks < block_seq_len:
                    # Check if this batch element needs to update the remainder block
                    if start_block <= full_blocks:
                        # Extract the remainder portion
                        remainder_data = partition[
                            b : b + 1, full_blocks * self.block_seq_stride :
                        ]
                        page_id = (
                            page_ids[b, full_blocks]
                            if full_blocks < page_ids.shape[1]
                            else page_ids[b, -1]
                        )

                        # Calculate index for the page table
                        block_index = (
                            page_id * page_stride
                            + transformer_block_index * transformer_block_stride
                            + idx
                        )

                        # Check if block_index is valid before indexing
                        if block_index < len(subblock_table):
                            # Get current block data
                            curr_values = subblock_table[block_index]

                            # If this is the start block and we have an offset, do selective update
                            if full_blocks == start_block and start_offset > 0:
                                # Reshape to access by position
                                curr_values = curr_values.reshape(
                                    self.block_seq_stride, -1, self.attn_head_dim
                                )

                                # Only update positions from start_offset onwards
                                # But only up to the remainder length
                                for pos in range(remainder):
                                    if pos >= start_offset:
                                        pos_in_remainder = pos - 0  # Adjust if needed
                                        if pos_in_remainder < remainder_data.shape[1]:
                                            curr_values[pos] = remainder_data[
                                                0, pos_in_remainder
                                            ]

                                # Flatten and write back
                                updated_flat = curr_values.reshape(-1)
                            else:
                                # Full update of remainder portion
                                # Pad remainder to block size
                                padded = torch.zeros_like(curr_values)

                                # Copy remainder data into padded tensor
                                flat_remainder = remainder_data.flatten()
                                padded[: flat_remainder.shape[0]] = flat_remainder
                                updated_flat = padded

                            # Write back to table
                            if subblock_table.dtype == torch.float8_e4m3fnuz:
                                subblock_table_as_int8 = subblock_table.view(
                                    dtype=torch.int8
                                )
                                updated_flat_int8 = updated_flat.view(dtype=torch.int8)
                                subblock_table_as_int8[block_index] = updated_flat_int8
                            else:
                                subblock_table[block_index] = updated_flat
