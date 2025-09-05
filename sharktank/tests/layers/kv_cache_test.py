# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from sharktank.layers import *
from sharktank.types import *
from sharktank.utils.testing import assert_tensor_close


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
    transformer_block_index = 1
    block_seq_stride = 4
    cache = PagedAttention(
        block_seq_stride=block_seq_stride,
        transformer_block_count=transformer_block_count,
        transformer_block_index=transformer_block_index,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_dtype=dtype,
        attn_dtype=dtype,
        use_rope=True,
        attention_chunk_size=None,
        device=None,
    )

    write_seq_length = seq_length - block_seq_stride
    page_count = bs * seq_length // block_seq_stride
    page_ids = torch.arange(page_count, dtype=torch.int64)
    page_ids = page_ids.view(bs, seq_length // block_seq_stride)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]

    allocation = cache.allocate(page_count=page_count)
    for t in allocation:
        t[...] = torch.full(t.shape, 0.0).to(dtype=dtype)

    # Write a prefill in:
    shape = bs, write_seq_length, attn_head_count, attn_head_dim
    write_ones = torch.rand(*shape).to(dtype=dtype)
    write_twos = torch.rand(*shape).to(dtype=dtype)

    cache.write(
        allocation,
        cache_partitions=[write_ones, write_twos],
        transformer_block_index=transformer_block_index,
        page_ids=write_page_ids,
    )

    read_back = cache.read(
        allocation,
        transformer_block_index=transformer_block_index,
        page_ids=write_page_ids,
    )
    assert_tensor_close(write_ones, read_back[0])
    assert_tensor_close(write_twos, read_back[1])

    # Check the others are still zero:
    for i in range(transformer_block_count):
        if i == transformer_block_index:
            continue
        read_ones = cache.read(
            allocation,
            transformer_block_index=i,
            page_ids=write_page_ids,
        )
        assert_tensor_close(
            read_ones[0], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )
        assert_tensor_close(
            read_ones[1], torch.full(read_ones[0].shape, 0.0).to(dtype=dtype)
        )

    # Write timestep
    ts_shape = (bs, 1, attn_head_count, attn_head_dim)
    write_threes = torch.rand(*ts_shape).to(dtype=dtype)
    write_fours = torch.rand(*ts_shape).to(dtype=dtype)

    for i in range(block_seq_stride):
        write_pos = torch.full((bs,), write_seq_length + i, dtype=torch.int64)
        cache.write_timestep(
            allocation,
            cache_partitions=[write_threes, write_fours],
            transformer_block_index=transformer_block_index,
            seq_positions=write_pos,
            page_ids=page_ids,
        )

    read_back = cache.read(
        allocation,
        transformer_block_index=transformer_block_index,
        page_ids=page_ids,
    )

    if dtype == torch.float8_e4m3fnuz:
        check_concat_0 = torch.concat(
            [write_ones.view(torch.int8)]
            + [write_threes.view(torch.int8)] * block_seq_stride,
            dim=1,
        ).view(torch.float8_e4m3fnuz)
        check_concat_1 = torch.concat(
            [write_twos.view(torch.int8)]
            + [write_fours.view(torch.int8)] * block_seq_stride,
            dim=1,
        ).view(torch.float8_e4m3fnuz)
    else:
        check_concat_0 = torch.concat(
            [write_ones] + [write_threes] * block_seq_stride, dim=1
        )
        check_concat_1 = torch.concat(
            [write_twos] + [write_fours] * block_seq_stride, dim=1
        )

    assert_tensor_close(check_concat_0, read_back[0])
    assert_tensor_close(check_concat_1, read_back[1])
