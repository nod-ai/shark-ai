# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from sharktank.layers import PagedAttention
import torch
from sharktank.utils import iterables_equal
from copy import deepcopy
from typing import List, Tuple
from sharktank import ops
from sharktank.types import SplitPrimitiveTensor


class PipelinedShardedPagedAttentionTest(unittest.TestCase):
    """Verify that the pipelined sharded paged attention behaves the same as the unpipelined unsharded variant."""

    def setUp(self):
        torch.manual_seed(12345)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.shard_count = 2
        self.pipeline_count = 2
        self.transformer_block_count = 5
        self.attn_head_count = self.shard_count * 7
        self.block_seq_stride = 19
        self.attn_head_dim = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.batch_size = 11
        self.block_seq_len = 2
        self.max_seq_len = self.block_seq_len * self.block_seq_stride

        block_to_device_lookup = []
        for block in range(self.transformer_block_count):
            pp_group = int(block * self.pipeline_count / self.transformer_block_count)
            zero_4_group = self.shard_count * pp_group
            block_to_device_lookup.append(
                tuple(i + zero_4_group for i in range(self.shard_count))
            )
        self.block_to_device_lookup = tuple(block_to_device_lookup)

        self.cache = PagedAttention(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.attn_head_count,
            block_seq_stride=self.block_seq_stride,
            attn_head_dim=self.attn_head_dim,
            cache_partition_count=self.cache_partition_count,
            cache_dtype=self.dtype,
            attn_dtype=self.dtype,
        )
        self.pipelined_sharded_cache = PagedAttention(
            shard_count=self.shard_count,
            block_to_device_lookup=self.block_to_device_lookup,
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.attn_head_count,
            block_seq_stride=self.block_seq_stride,
            attn_head_dim=self.attn_head_dim,
            cache_partition_count=self.cache_partition_count,
            cache_dtype=self.dtype,
            attn_dtype=self.dtype,
        )

    def make_unsharded_and_sharded_equal_cache_states(
        self,
    ) -> Tuple[List[torch.Tensor], List[SplitPrimitiveTensor]]:
        cache_state = self.cache.allocate(self.page_count)
        cache_state[0] = torch.rand_like(cache_state[0])
        sharded_cache_state = self.pipelined_sharded_cache.shard_state(
            deepcopy(cache_state)
        )
        self.assert_equal_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )
        return cache_state, sharded_cache_state

    def assert_equal_unsharded_and_sharded_cache_states(
        self,
        cache_state: List[torch.Tensor],
        pipelined_sharded_cache_state: List[SplitPrimitiveTensor],
    ):
        pipelined_sharded_states_as_unsharded = [
            ops.unshard(unflatted_page).flatten(start_dim=1)
            for unflatted_page in self.pipelined_sharded_cache.unflatten_page_tables(
                pipelined_sharded_cache_state
            )
        ]
        pipelined_sharded_states_as_single = ops.cat(
            pipelined_sharded_states_as_unsharded, dim=1
        )
        assert iterables_equal(
            cache_state[0].shape, pipelined_sharded_states_as_single.shape
        )
        assert ops.equal(
            cache_state[0],
            pipelined_sharded_states_as_single,
        )

    def testAllocate(self):
        cache_state = self.cache.allocate(self.page_count)
        pipelined_sharded_cache_allocation = self.pipelined_sharded_cache.allocate(
            self.page_count
        )
        assert len(cache_state) == 1
        assert len(pipelined_sharded_cache_allocation) == self.pipeline_count
        assert all(t.shape[0] == self.page_count for t in cache_state)
        assert cache_state[0].shape[1] == sum(
            t.shape[1] for t in pipelined_sharded_cache_allocation
        )
        assert all(t.shard_dim == 1 for t in pipelined_sharded_cache_allocation)
        assert all(
            t.shard_count == self.shard_count
            for t in pipelined_sharded_cache_allocation
        )

    def testUnflattenPageTable(self):
        cache_state = self.cache.allocate(self.page_count)
        assert len(cache_state) == 1
        pipelined_sharded_cache_state = self.pipelined_sharded_cache.allocate(
            self.page_count
        )

        unflattened_state = self.cache.unflatten_page_tables(cache_state)
        pipelined_sharded_unflattened_state = (
            self.pipelined_sharded_cache.unflatten_page_tables(
                pipelined_sharded_cache_state
            )
        )
        # [0] is page count
        assert all(
            sharded_page_slab.shape[0] == self.page_count
            for sharded_page_slab in pipelined_sharded_unflattened_state
        )
        # [1] is for block count, and split across pipelines
        assert unflattened_state[0].shape[1] == self.transformer_block_count
        assert (
            sum(page_slab.shape[1] for page_slab in pipelined_sharded_unflattened_state)
            == self.transformer_block_count
        )
        # [2:] should be the same
        assert all(
            iterables_equal(page_slab.shape[2:], unflattened_state[0].shape[2:])
            for page_slab in pipelined_sharded_unflattened_state
        )
        assert all(
            sharded_page_slab.shard_dim == 4
            for sharded_page_slab in pipelined_sharded_unflattened_state
        )
        assert all(
            sharded_page_slab.shard_count == self.shard_count
            for sharded_page_slab in pipelined_sharded_unflattened_state
        )

    def testRead(self):
        (
            cache_state,
            sharded_cache_state,
        ) = self.make_unsharded_and_sharded_equal_cache_states()

        transformer_block_index = 1
        page_ids = torch.randint(
            low=0, high=self.page_count, size=[self.batch_size, self.block_seq_len]
        ).reshape([self.batch_size, self.block_seq_len])
        unsharded_read = self.cache.read(
            state=cache_state,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
            seq_len=self.block_seq_len * self.block_seq_stride,
        )
        sharded_page_ids = ops.replicate(page_ids, count=self.shard_count)
        sharded_read = self.pipelined_sharded_cache.read(
            state=sharded_cache_state,
            transformer_block_index=transformer_block_index,
            page_ids=sharded_page_ids,
            seq_len=self.block_seq_len * self.block_seq_stride,
        )
        for unsharded, sharded in zip(unsharded_read, sharded_read):
            assert ops.equal(unsharded, ops.unshard(sharded))

    def testWriteTimestep(self):
        (
            cache_state,
            sharded_cache_state,
        ) = self.make_unsharded_and_sharded_equal_cache_states()

        cache_partitions = [
            torch.rand(
                self.batch_size,
                1,
                self.attn_head_count,
                self.attn_head_dim,
            )
            for _ in range(self.cache_partition_count)
        ]
        transformer_block_index = 1
        seq_positions = torch.randint(
            low=0, high=self.max_seq_len, size=[self.batch_size]
        )
        page_ids = torch.randperm(self.batch_size * self.block_seq_len).reshape(
            [self.batch_size, self.block_seq_len]
        )
        self.cache.write_timestep(
            state=cache_state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )
        sharded_cache_partitions = deepcopy(
            [
                ops.reshard_split(t, dim=2, count=self.shard_count)
                for t in cache_partitions
            ]
        )
        sharded_seq_positions = ops.replicate(seq_positions, count=self.shard_count)
        sharded_page_ids = ops.replicate(page_ids, count=self.shard_count)
        self.pipelined_sharded_cache.write_timestep(
            state=sharded_cache_state,
            cache_partitions=sharded_cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=sharded_seq_positions,
            page_ids=sharded_page_ids,
        )
        self.assert_equal_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )

    def testWrite(self):
        (
            cache_state,
            sharded_cache_state,
        ) = self.make_unsharded_and_sharded_equal_cache_states()

        cache_partitions = [
            torch.rand(
                self.batch_size,
                self.block_seq_len * self.block_seq_stride,
                self.attn_head_count,
                self.attn_head_dim,
            )
            for _ in range(self.cache_partition_count)
        ]
        transformer_block_index = 1
        assert self.batch_size * self.block_seq_len <= self.page_count
        page_ids = torch.randperm(self.batch_size * self.block_seq_len).reshape(
            [self.batch_size, self.block_seq_len]
        )
        self.cache.write(
            state=cache_state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
        )
        sharded_cache_partitions = deepcopy(
            [
                ops.reshard_split(t, dim=2, count=self.shard_count)
                for t in cache_partitions
            ]
        )
        sharded_page_ids = ops.replicate(page_ids, count=self.shard_count)
        self.pipelined_sharded_cache.write(
            state=sharded_cache_state,
            cache_partitions=sharded_cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=sharded_page_ids,
        )
        self.assert_equal_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )
