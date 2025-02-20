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
            self.block_seq_stride,
            self.attn_head_count // self.shard_count,
            self.attn_head_dim,
        ]

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

        def read_cache_partition(index: int):
            subblock_ids = base_subblock_ids + index
            # TODO: Potentially clamp all page 0 indices to the mask value.
            # Or even better, require that the ids are replicated such that access is
            # legal.
            # Now for each of the k/v attn_block_ids, which have been adjusted to
            # index into the sub-pages, we flatten to do a linear index_select
            # copy of the sub-blocks by collapsing the first two dims so we have
            # a linear list.
            selected = (
                ops.index_select(subblock_table, 0, subblock_ids.flatten(0, 1))
                .unflatten(0, blocked_shape[0:2])
                .flatten(1, 2)
            )
            return selected

        key = read_cache_partition(0)
        value = read_cache_partition(1)

        return key[:, :seq_len], value[:, :seq_len]

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

            indices = (page_id, transformer_block, partitions, page_offset)
            values = ops.to(cache_partition, dtype=page_table.dtype)
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=indices, values=values_int8)
            else:
                page_table.index_put_(indices=indices, values=values)

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
            part_block_view = partition.unflatten(
                1, (block_seq_len, self.block_seq_stride)
            )
            part_block_view = part_block_view.flatten(0, 1)

            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            ).flatten(0, 1)

            part_block = ops.to(part_block_view, dtype=subblock_table.dtype)
            if subblock_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                subblock_table_as_int8 = subblock_table.view(dtype=torch.int8)
                part_block_as_int8 = part_block.view(dtype=torch.int8)
                subblock_table_as_int8.index_copy_(0, subblock_ids, part_block_as_int8)
            else:
                subblock_table.index_copy_(0, subblock_ids, part_block)
