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
    * attention heads
    * block sequence stride (number of sequence positions per block)
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
        block_seq_stride: int = 16,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        shard_count: int = 1,
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count
        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        # Some derived values based on attributes.
        self.sub_page_dims = [
            self.transformer_block_count,
            2,
            self.attn_head_count // self.shard_count,
            self.block_seq_stride,
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
            return SplitPrimitiveTensor(ts=shards, shard_dim=3)

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
                2,
                self.attn_head_count,
                self.block_seq_stride,
                self.attn_head_dim,
            ]
        )
        sharded_page_table = ops.reshard_split(
            page_table, dim=3, count=self.shard_count
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

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        page_stride = self.transformer_block_count * 2
        transformer_block_stride = 2
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
                # Read Layout is: (bs, block_seq_len), kv_head_count, block_seq_stride, head_dim
                # Output Layout is: bs, (block_seq_len, block_seq_stride), kv_head_count, head_dim
                ops.index_select(subblock_table, 0, subblock_ids.flatten(0, 1))
                # bs, block_seq_len, kv_head_count, block_seq_stride, head_dim
                .unflatten(0, (bs, block_seq_len))
                # bs, block_seq_len, block_seq_stride, kv_head_count, head_dim
                .transpose(2, 3)
                # bs, (block_seq_len, block_seq_stride), kv_head_count, head_dim
                .flatten(1, 2)
            )
            return selected

        key = read_cache_partition(0)
        value = read_cache_partition(1)

        return key[:, :seq_len], value[:, :seq_len]

    def write_timestep(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        # [bs, 1, attn_head_count, attn_head_dim]
        key: Union[torch.Tensor, SplitPrimitiveTensor],
        # [bs, 1, attn_head_count, attn_head_dim]
        value: Union[torch.Tensor, SplitPrimitiveTensor],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, max_seqlen // block_pos_stride]
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a single batched timestep across all cache partitions."""
        device = self.device
        page_table = self.unflatten_page_table(state)  # 6D
        bs, *_ = seq_positions.shape

        page_index = seq_positions // self.block_seq_stride
        page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
        page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

        # This probably be written much better..., but after fighting
        # a lot of with torch-mlir and dynamo, this is a hacky enough
        # version that works. We are doing a lot of praying to the compiler
        # here for scatter fusion.
        # vLLM does this write using a custom kernel to write the key and
        # value partitions.
        #
        # Here, we are trying to get the index to be a broadcasted version
        # of the partition layout in which we are going to write in:
        # [bs, kv_head_count, 1]
        page_id = page_id.unsqueeze(-1).expand(bs, self.attn_head_count, 1)
        head_id = (
            torch.arange(0, self.attn_head_count, dtype=page_ids.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, self.attn_head_count, 1)
        )
        page_offset = page_offset.unsqueeze(-1).expand(bs, self.attn_head_count, 1)

        # This is a hack. Without this IREE generates a seperate dispatch for
        # each layer, as well as fails to compiles for all dispatches except
        # for layer 0.
        block_idx = torch.full(
            (bs, self.attn_head_count, 1), transformer_block_index, dtype=page_ids.dtype
        )

        for idx, cache_partition in enumerate([key, value]):
            # Input Layout: bs, 1, kv_head_count, attn_head_dim
            # Partition Layout: (bs, 1), kv_head_count, block_seq_stride, head_dim

            # Same hack as above.
            part_idx = torch.full(
                (bs, self.attn_head_count, 1), idx, dtype=page_ids.dtype
            )

            # bs, kv_head_count, 1, attn_head_dim
            partition_view = cache_partition.transpose(1, 2)
            page_table[
                page_id, block_idx, part_idx, head_id, page_offset
            ] = partition_view

    def write(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        key: Union[torch.Tensor, SplitPrimitiveTensor],
        value: Union[torch.Tensor, SplitPrimitiveTensor],
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
        page_stride = self.transformer_block_count * 2
        transformer_block_stride = 2
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        def reshape_input(x):
            # Input Layout: bs, (block_seq_len, block_seq_stride), kv_head_count, head_dim
            # Write Layout: (bs, block_seq_len), kv_head_count, block_seq_stride, head_dim

            return (
                x
                # bs, block_seq_len, block_seq_stride, kv_head_count, head_dim
                .unflatten(1, (block_seq_len, self.block_seq_stride))
                # bs, block_seq_len, kv_head_count, block_seq_stride, head_dim
                .transpose(2, 3)
                # (bs, block_seq_len), kv_head_count, block_seq_stride, head_dim
                .flatten(0, 1)
            )

        key_ids = base_subblock_ids.flatten(0, 1)
        value_ids = base_subblock_ids.flatten(0, 1) + 1

        subblock_table.index_copy_(0, key_ids, reshape_input(key))
        subblock_table.index_copy_(0, value_ids, reshape_input(value))
