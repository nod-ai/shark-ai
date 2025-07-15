# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import torch

from sharktank.layers.rotary_embedding import build_rotary_layer
from sharktank.utils.testing import assert_tensor_close
from sharktank import ops
from sharktank.types import (
    ShardedTensor,
    SplitPrimitiveTensor,
    unbox_tensor,
)

import unittest
from typing import List, Optional
import os


def test_sharded_rotary_table():
    bs = 4
    rope_dims = 16
    heads = 8
    max_seqlen = 128
    rope_freq_base = None

    # First we setup and get the default rotary embedding layer
    xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    xk = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    default_layer = build_rotary_layer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
    )
    oq = default_layer(xt=xq, start_index=0)
    ok = default_layer(xt=xk, start_index=0)

    # Then we can shard the same inputs and layer
    xq = SplitPrimitiveTensor(ts=xq, shard_dim=2, shard_count=4)
    xk = SplitPrimitiveTensor(ts=xk, shard_dim=2, shard_count=4)
    shard_layer = build_rotary_layer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
        tensor_parallelism_size=4,
    )
    sq = shard_layer(xt=xq, start_index=0)
    sk = shard_layer(xt=xk, start_index=0)

    # Gathering and unboxing should yield the same results
    sq = ops.unshard(sq)
    sk = ops.unshard(sk)

    assert_tensor_close(sq, oq)
    assert_tensor_close(sk, ok)
