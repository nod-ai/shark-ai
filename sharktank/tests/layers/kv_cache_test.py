# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from sharktank.ops import replicate, reshard_split, unshard
from sharktank.layers import *
from sharktank.types import *


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fnuz,
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ],
)
def test_paged(dtype: torch.dtype):
    bs = 4
    seq_length = 24
    attn_head_count = 4
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4
    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        dtype=dtype,
        device=None,
    )

    write_seq_length = seq_length - 4
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64)
    page_ids = page_ids.view(bs, seq_length // block_seq_stride)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]

    allocation = cache.allocate(page_count=page_count)
    for t in allocation:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Write a prefill in:
    write_ones = torch.full(
        (bs, write_seq_length, attn_head_count, attn_head_dim),
        1.0,
    ).to(dtype=dtype)
    write_twos = torch.full(
        (bs, write_seq_length, attn_head_count, attn_head_dim),
        2.0,
    ).to(dtype=dtype)

    cache.write(
        allocation,
        cache_partitions=[write_ones, write_twos],
        transformer_block_index=1,
        page_ids=write_page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        seq_len=write_seq_length,
        page_ids=write_page_ids,
    )
    torch.testing.assert_close(write_ones, read_back[0])
    torch.testing.assert_close(write_twos, read_back[1])

    # Check the others are still zero:
    for i in range(transformer_block_count):
        if i == 1:
            continue
        read_ones = cache.read(
            allocation,
            transformer_block_index=i,
            seq_len=write_seq_length,
            page_ids=write_page_ids,
        )
        torch.testing.assert_close(
            read_ones[0], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )
        torch.testing.assert_close(
            read_ones[1], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )

    # Write timestep
    write_threes = torch.full((bs, 1, attn_head_count, attn_head_dim), 3.0).to(
        dtype=dtype
    )
    write_fours = torch.full((bs, 1, attn_head_count, attn_head_dim), 4.0).to(
        dtype=dtype
    )
    write_pos = torch.full((bs,), write_seq_length, dtype=torch.int64)
    cache.write_timestep(
        allocation,
        cache_partitions=[write_threes, write_fours],
        transformer_block_index=1,
        seq_positions=write_pos,
        page_ids=page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        seq_len=write_seq_length + 1,
        page_ids=page_ids,
    )

    if dtype == torch.float8_e4m3fnuz:
        check_concat_0 = torch.concat(
            [write_ones.view(torch.int8), write_threes.view(torch.int8)], dim=1
        ).view(torch.float8_e4m3fnuz)
        check_concat_1 = torch.concat(
            [write_twos.view(torch.int8), write_fours.view(torch.int8)], dim=1
        ).view(torch.float8_e4m3fnuz)
    else:
        check_concat_0 = torch.concat([write_ones, write_threes], dim=1)
        check_concat_1 = torch.concat([write_twos, write_fours], dim=1)

    torch.testing.assert_close(check_concat_0, read_back[0])
    torch.testing.assert_close(check_concat_1, read_back[1])


def test_sharded_paged():
    bs = 4
    seq_length = 24
    attn_head_count = 8
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4
    shard_count = 4
    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        shard_count=shard_count,
        dtype=torch.float32,
        device=None,
    )

    write_seq_length = seq_length - 4
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64)
    page_ids = page_ids.view(bs, seq_length // block_seq_stride)
    page_ids = replicate(page_ids, shard_count)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]

    allocation = cache.allocate(page_count=page_count)

    # Write a prefill in:
    write_ones = reshard_split(
        torch.full(
            (bs, write_seq_length, attn_head_count, attn_head_dim),
            1.0,
            dtype=torch.float32,
        ),
        dim=2,
        count=shard_count,
    )
    write_twos = reshard_split(
        torch.full(
            (bs, write_seq_length, attn_head_count, attn_head_dim),
            2.0,
            dtype=torch.float32,
        ),
        dim=2,
        count=shard_count,
    )

    cache.write(
        allocation,
        cache_partitions=[write_ones, write_twos],
        transformer_block_index=1,
        page_ids=write_page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        seq_len=write_seq_length,
        page_ids=write_page_ids,
    )
    torch.testing.assert_close(unshard(write_ones), unshard(read_back[0]))
    torch.testing.assert_close(unshard(write_twos), unshard(read_back[1]))

    # Write timestep
    write_threes = reshard_split(
        torch.full((bs, 1, attn_head_count, attn_head_dim), 3.0, dtype=torch.float32),
        dim=2,
        count=shard_count,
    )

    write_fours = reshard_split(
        torch.full((bs, 1, attn_head_count, attn_head_dim), 4.0, dtype=torch.float32),
        dim=2,
        count=shard_count,
    )

    write_pos = replicate(
        torch.full((bs,), write_seq_length, dtype=torch.int64), shard_count
    )

    cache.write_timestep(
        allocation,
        cache_partitions=[write_threes, write_fours],
        transformer_block_index=1,
        seq_positions=write_pos,
        page_ids=page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=1,
        seq_len=write_seq_length + 1,
        page_ids=page_ids,
    )

    check_concat_0 = torch.concat([unshard(write_ones), unshard(write_threes)], dim=1)
    check_concat_1 = torch.concat([unshard(write_twos), unshard(write_fours)], dim=1)

    torch.testing.assert_close(check_concat_0, unshard(read_back[0]))
    torch.testing.assert_close(check_concat_1, unshard(read_back[1]))
