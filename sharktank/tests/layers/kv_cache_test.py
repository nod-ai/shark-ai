# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
from parameterized import parameterized_class
import pytest
import torch
import unittest

from amdsharktank.layers import *
from amdsharktank.layers.paged_attention import build_cache
from amdsharktank.types import *
from amdsharktank.utils.testing import assert_tensor_close

import amdsharktank.ops as ops

_DTYPES = [
    torch.float8_e4m3fnuz,
    torch.bfloat16,
    torch.float16,
    torch.float32,
]

_PP = [1, 4, 8]


@parameterized_class(
    ("dtype", "pp"), [(dtype, pp) for dtype, pp in itertools.product(_DTYPES, _PP)]
)
class PagedKVCacheTest(unittest.TestCase):
    def setUp(self):
        self.bs = 4
        self.seq_length = 24
        self.attn_head_count = 4
        self.attn_head_dim = 16
        self.transformer_block_count = 8
        self.transformer_block_index = 3
        self.block_seq_stride = 4
        self.write_seq_length = self.seq_length - self.block_seq_stride
        self.page_count = self.bs * self.seq_length // self.block_seq_stride

        self.parallelism_config = ParallelismConfig.default_config(
            block_count=self.transformer_block_count, pp=self.pp
        )
        self.kv_cache = build_cache(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.attn_head_count,
            attn_head_dim=self.attn_head_dim,
            block_seq_stride=self.block_seq_stride,
            cache_dtype=self.dtype,
            parallelism_config=self.parallelism_config,
        )
        self.cache = PagedGQAttention(
            kv_cache=self.kv_cache,
            transformer_block_index=self.transformer_block_index,
            attn_dtype=self.dtype,
            use_rope=True,
            attention_chunk_size=None,
        )

    def parallize_tensors_if_needed(
        self, *tensors: torch.Tensor
    ) -> list[torch.Tensor | ReplicatedTensor]:
        if self.parallelism_config.pipeline_size == 1:
            if len(tensors) == 1:
                return tensors[0]
            return list(tensors)
        pipeline = self.parallelism_config.pipeline_for_block(
            self.transformer_block_index
        )
        devices = self.parallelism_config.devices_for_pipeline(pipeline)

        new_tensors = []
        for tensor in tensors:
            new_tensors.append(
                ReplicatedTensor(ts=tensor, shard_count=len(devices), devices=devices)
            )

        if len(new_tensors) == 1:
            return new_tensors[0]
        return new_tensors

    def get_paged_ids(
        self,
    ) -> tuple[torch.Tensor | ReplicatedTensor, torch.Tensor | ReplicatedTensor]:
        page_ids = torch.arange(self.page_count, dtype=torch.int64)
        page_ids = page_ids.view(self.bs, self.seq_length // self.block_seq_stride)
        write_page_ids = page_ids[:, : self.write_seq_length // self.block_seq_stride]

        page_ids, write_page_ids = self.parallize_tensors_if_needed(
            page_ids, write_page_ids
        )
        return page_ids, write_page_ids

    def get_zeroed_allocation(self) -> CacheAllocation:
        allocation = self.cache.allocate(page_count=self.page_count)
        for t in allocation:
            t[...] = torch.full(t.shape, 0.0).to(dtype=self.dtype)
        return allocation

    def get_rand_content_to_write(
        self, shape: tuple[int, ...]
    ) -> tuple[torch.Tensor | ReplicatedTensor, torch.Tensor | ReplicatedTensor]:
        write_ones = torch.rand(*shape).to(dtype=self.dtype)
        write_twos = torch.rand(*shape).to(dtype=self.dtype)

        write_ones, write_twos = self.parallize_tensors_if_needed(
            write_ones, write_twos
        )
        return write_ones, write_twos

    def test_paged(self):
        page_ids, write_page_ids = self.get_paged_ids()
        allocation = self.get_zeroed_allocation()

        # Write a prefill in:
        shape = self.bs, self.write_seq_length, self.attn_head_count, self.attn_head_dim
        write_ones, write_twos = self.get_rand_content_to_write(shape)
        self.cache.write(
            allocation,
            cache_partitions=[write_ones, write_twos],
            transformer_block_index=self.transformer_block_index,
            page_ids=write_page_ids,
        )

        read_back = self.cache.read(
            allocation,
            transformer_block_index=self.transformer_block_index,
            page_ids=write_page_ids,
        )
        assert_tensor_close(write_ones, read_back[0])
        assert_tensor_close(write_twos, read_back[1])

        # Check the others are still zero:
        for i in range(self.transformer_block_count):
            if i == self.transformer_block_index:
                continue
            read_one, read_two = self.cache.read(
                allocation,
                transformer_block_index=i,
                page_ids=write_page_ids,
            )
            assert_tensor_close(
                read_one, torch.full(read_one.shape, 0.0).to(dtype=self.dtype)
            )
            assert_tensor_close(
                read_two, torch.full(read_two.shape, 0.0).to(dtype=self.dtype)
            )

        # Write timestep
        ts_shape = (self.bs, 1, self.attn_head_count, self.attn_head_dim)
        write_threes, write_fours = self.get_rand_content_to_write(ts_shape)
        for i in range(self.block_seq_stride):
            write_pos = torch.full(
                (self.bs,), self.write_seq_length + i, dtype=torch.int64
            )
            write_pos = self.parallize_tensors_if_needed(write_pos)
            self.cache.write_timestep(
                allocation,
                cache_partitions=[write_threes, write_fours],
                transformer_block_index=self.transformer_block_index,
                seq_positions=write_pos,
                page_ids=page_ids,
            )

        read_back = self.cache.read(
            allocation,
            transformer_block_index=self.transformer_block_index,
            page_ids=page_ids,
        )

        if self.dtype == torch.float8_e4m3fnuz:
            check_cat_0 = ops.cat(
                [write_ones.view(torch.int8)]
                + [write_threes.view(torch.int8)] * self.block_seq_stride,
                dim=1,
            ).view(torch.float8_e4m3fnuz)
            check_cat_1 = ops.cat(
                [write_twos.view(torch.int8)]
                + [write_fours.view(torch.int8)] * self.block_seq_stride,
                dim=1,
            ).view(torch.float8_e4m3fnuz)
        else:
            check_cat_0 = ops.cat(
                [write_ones] + [write_threes] * self.block_seq_stride, dim=1
            )
            check_cat_1 = ops.cat(
                [write_twos] + [write_fours] * self.block_seq_stride, dim=1
            )

        assert_tensor_close(check_cat_0, read_back[0])
        assert_tensor_close(check_cat_1, read_back[1])
