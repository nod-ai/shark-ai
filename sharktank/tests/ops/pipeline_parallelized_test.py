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

class TransferIfNeededTest(unittest.TestCase):
    def testTransferOnSameDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2+i for i in range(shard_count))
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        pre_1 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices, devices_pinned=True)
        pre_2 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices, devices_pinned=False)

        post_1, post_2 = ops.transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testTransferOnDifferentDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices_pinned = tuple(2+i for i in range(shard_count))
        devices_free = tuple(2 + 2*i for i in range(shard_count))
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        pre_1 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices_pinned, devices_pinned=True)
        pre_2 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices_free, devices_pinned=False)

        post_1, post_2 = ops.transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices_pinned):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testBothPinnedOnSameDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2+i for i in range(shard_count))
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        pre_1 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices, devices_pinned=True)
        pre_2 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices, devices_pinned=True)

        post_1, post_2 = ops.transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testBothPinnedOnDifferentDevices(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices_pinned = tuple(2+i for i in range(shard_count))
        devices_free = tuple(2 + 2*i for i in range(shard_count))
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        pre_1 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices_pinned, devices_pinned=True)
        pre_2 = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices_free, devices_pinned=True)

        try:
            ops.transfer_if_needed(pre_1, pre_2)
        except ValueError:
            return
        assert False  # Should have thrown a ValueError since both tensors are pinned, but devices are not the same
