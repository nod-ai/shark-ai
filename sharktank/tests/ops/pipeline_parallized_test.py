# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
from parameterized import parameterized

import torch

from sharktank import ops
from sharktank.types import *
from sharktank.types import sharding
from sharktank.layers import Conv2DLayer


class AllGatherTest(unittest.TestCase):
    def testAllGather(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32)
            for _ in range(shard_count)
        ]
        expected_result = torch.cat(shards, dim=shard_dim)

        devices = (0, 6, 1)
        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards, devices=devices, devices_pinned=True)
        actual_result = ops.all_gather(sharded)

        for i in range(shard_count):
            torch.testing.assert_close(actual_result.shards[i].as_torch(), expected_result)
            assert actual_result.devices[i] == devices[i]

class AllReduceTest(unittest.TestCase):
    def testAllReduce(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32)
            for _ in range(shard_count)
        ]
        expected_result = torch.add(torch.add(shards[0], shards[1]), shards[2])

        devices = (0, 6, 1)
        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards, devices=devices, devices_pinned=True)
        actual_result = ops.all_reduce(sharded)

        for i in range(shard_count):
            torch.testing.assert_close(actual_result.shards[i].as_torch(), expected_result)
            assert actual_result.devices[i] == devices[i]
