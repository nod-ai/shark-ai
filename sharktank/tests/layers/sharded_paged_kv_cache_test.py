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


class ShardedPagedKVCacheTest(unittest.TestCase):
    """Verify that the sharded paged KV cache behaves as the unsharded variant."""

    def setUp(self):
        torch.manual_seed(12345)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.shard_count = 3
        self.transformer_block_count = 5
        self.attn_head_count = self.shard_count * 7
        self.block_seq_stride = 19
        self.attn_head_dim = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.batch_size = 11
        self.block_seq_len = 2
        self.max_seq_len = self.block_seq_len * self.block_seq_stride

        self.cache = PagedAttention(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.attn_head_count,
            block_seq_stride=self.block_seq_stride,
            attn_head_dim=self.attn_head_dim,
            cache_partition_count=self.cache_partition_count,
            cache_dtype=self.dtype,
            attn_dtype=self.dtype,
        )
        self.sharded_cache = PagedAttention(
            shard_count=self.shard_count,
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
        sharded_cache_state = self.sharded_cache.shard_state(deepcopy(cache_state))
        self.assert_equal_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )
        return cache_state, sharded_cache_state

    def assert_equal_unsharded_and_sharded_cache_states(
        self,
        cache_state: List[torch.Tensor],
        sharded_cache_state: List[SplitPrimitiveTensor],
    ):
        sharded_state_as_unsharded = ops.unshard(
            self.sharded_cache.unflatten_page_table(sharded_cache_state)
        ).flatten(start_dim=1)
        assert ops.equal(
            cache_state[0],
            sharded_state_as_unsharded,
        )

    def testAllocate(self):
        cache_state = self.cache.allocate(self.page_count)
        sharded_cache_state = self.sharded_cache.allocate(self.page_count)
        assert len(cache_state) == 1
        assert len(sharded_cache_state) == 1
        assert iterables_equal(cache_state[0].shape, sharded_cache_state[0].shape)
        assert sharded_cache_state[0].shard_dim == 1
        assert sharded_cache_state[0].shard_count == self.shard_count

    def testUnflattenPageTable(self):
        cache_state = self.cache.allocate(self.page_count)
        sharded_cache_state = self.sharded_cache.allocate(self.page_count)

        unflattened_cache_state = self.cache.unflatten_page_table(cache_state)
        sharded_unflattened_cache_state = self.sharded_cache.unflatten_page_table(
            sharded_cache_state
        )
        assert iterables_equal(
            unflattened_cache_state.shape, sharded_unflattened_cache_state.shape
        )
        assert sharded_unflattened_cache_state.shard_dim == 4
        assert sharded_unflattened_cache_state.shard_count == self.shard_count
        assert sharded_unflattened_cache_state.shape[0] == self.page_count

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
        sharded_read = self.sharded_cache.read(
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
        self.sharded_cache.write_timestep(
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
        self.sharded_cache.write(
            state=sharded_cache_state,
            cache_partitions=sharded_cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=sharded_page_ids,
        )
        self.assert_equal_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )
