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


class PipelinedPagedAttentionTest(unittest.TestCase):
    """Verify that the pipelined paged attention behaves the same as the unpipelined variant."""

    def setUp(self):
        torch.manual_seed(12345)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.shard_count = 1
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
        self.pipelined_cache = PagedAttention(
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

    def make_unpipelined_and_pipelined_equal_cache_states(
        self,
    ) -> Tuple[List[torch.Tensor], List[SplitPrimitiveTensor]]:
        cache_state = self.cache.allocate(self.page_count)
        cache_state[0] = torch.rand_like(cache_state[0])
        pipelined_cache_state = self.pipelined_cache.shard_state(deepcopy(cache_state))
        self.assert_equal_unpipelined_and_pipelined_cache_states(
            cache_state, pipelined_cache_state
        )
        return cache_state, pipelined_cache_state

    def assert_equal_unpipelined_and_pipelined_cache_states(
        self,
        cache_state: List[torch.Tensor],
        pipelined_cache_state: List[SplitPrimitiveTensor],
    ):
        pipelined_states_unreplicated = [
            ops.unshard(unflatted_page).flatten(start_dim=1)
            for unflatted_page in self.pipelined_cache.unflatten_page_tables(
                pipelined_cache_state
            )
        ]
        pipelined_states_as_single = ops.cat(pipelined_states_unreplicated, dim=1)
        assert iterables_equal(cache_state[0].shape, pipelined_states_as_single.shape)
        assert ops.equal(
            cache_state[0],
            pipelined_states_as_single,
        )

    def testAllocate(self):
        cache_state = self.cache.allocate(self.page_count)
        pipelined_cache_allocation = self.pipelined_cache.allocate(self.page_count)
        assert len(cache_state) == 1
        assert len(pipelined_cache_allocation) == self.pipeline_count
        assert all(t.shape[0] == self.page_count for t in cache_state)
        assert cache_state[0].shape[1] == sum(
            t.shape[1] for t in pipelined_cache_allocation
        )

    def testUnflattenPageTable(self):
        cache_state = self.cache.allocate(self.page_count)
        assert len(cache_state) == 1
        pipelined_cache_state = self.pipelined_cache.allocate(self.page_count)

        unflattened_state = self.cache.unflatten_page_tables(cache_state)
        pipelined_unflattened_state = self.pipelined_cache.unflatten_page_tables(
            pipelined_cache_state
        )
        # [0] is page count
        assert all(
            pipelined_page_slab.shape[0] == self.page_count
            for pipelined_page_slab in pipelined_unflattened_state
        )
        # [1] is for block count, and split across pipelines
        assert unflattened_state[0].shape[1] == self.transformer_block_count
        assert (
            sum(page_slab.shape[1] for page_slab in pipelined_unflattened_state)
            == self.transformer_block_count
        )
        # [2:] should be the same
        assert all(
            iterables_equal(page_slab.shape[2:], unflattened_state[0].shape[2:])
            for page_slab in pipelined_unflattened_state
        )

    def testRead(self):
        (
            cache_state,
            pipelined_cache_state,
        ) = self.make_unpipelined_and_pipelined_equal_cache_states()

        transformer_block_index = 1
        page_ids = torch.randint(
            low=0, high=self.page_count, size=[self.batch_size, self.block_seq_len]
        ).reshape([self.batch_size, self.block_seq_len])
        unpipelined_read = self.cache.read(
            state=cache_state,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
        )
        pipelined_page_ids = ops.replicate(page_ids, count=self.shard_count)
        pipelined_read = self.pipelined_cache.read(
            state=pipelined_cache_state,
            transformer_block_index=transformer_block_index,
            page_ids=pipelined_page_ids,
        )
        for unpipelined, pipelined in zip(unpipelined_read, pipelined_read):
            assert ops.equal(unpipelined, ops.unshard(pipelined))

    def testWriteTimestep(self):
        (
            cache_state,
            pipelined_cache_state,
        ) = self.make_unpipelined_and_pipelined_equal_cache_states()

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
        pipelined_cache_partitions = deepcopy(
            [ops.replicate(t, count=self.shard_count) for t in cache_partitions]
        )
        pipelined_seq_positions = ops.replicate(seq_positions, count=self.shard_count)
        pipelined_page_ids = ops.replicate(page_ids, count=self.shard_count)
        self.pipelined_cache.write_timestep(
            state=pipelined_cache_state,
            cache_partitions=pipelined_cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=pipelined_seq_positions,
            page_ids=pipelined_page_ids,
        )
        self.assert_equal_unpipelined_and_pipelined_cache_states(
            cache_state, pipelined_cache_state
        )

    def testWrite(self):
        (
            cache_state,
            pipelined_cache_state,
        ) = self.make_unpipelined_and_pipelined_equal_cache_states()

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
        pipelined_cache_partitions = deepcopy(
            [ops.replicate(t, count=self.shard_count) for t in cache_partitions]
        )
        pipelined_page_ids = ops.replicate(page_ids, count=self.shard_count)
        self.pipelined_cache.write(
            state=pipelined_cache_state,
            cache_partitions=pipelined_cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=pipelined_page_ids,
        )
        self.assert_equal_unpipelined_and_pipelined_cache_states(
            cache_state, pipelined_cache_state
        )
