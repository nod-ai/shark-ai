# Copyright 2025 Advanced Micro Devices, Inc.
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
    
    def testMultiTensorsNoPinned(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        t_pre = [
            SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=tuple(shard_count*i + d for d in range(shard_count)), devices_pinned=False)
            for i in range(tensor_count)
        ]
        
        try:
            ops.transfer_if_needed(*t_pre)
        except ValueError:
            return
        assert False # Should have thrown a ValueError since no devices are different but none are pinned

    def testMultiTensorsOnePinned(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        t_pre = [
            SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=tuple(shard_count*i + d for d in range(shard_count)), devices_pinned=(i==0))
            for i in range(tensor_count)
        ]
        t_post = ops.transfer_if_needed(*t_pre)

        for i in range(tensor_count):
            assert all(d_pre == d_post for d_pre, d_post in zip(t_pre[0].devices, t_post[i].devices))

    def testMultiTensorsMultiPinnedNoConflict(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        t_pre = [
            SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=tuple(shard_count*i*(i % 2 != 0) + d for d in range(shard_count)), devices_pinned=(i % 2 == 0))
            for i in range(tensor_count)
        ]
        t_post = ops.transfer_if_needed(*t_pre)

        for i in range(tensor_count):
            assert all(d_pre == d_post for d_pre, d_post in zip(t_pre[0].devices, t_post[i].devices))

    def testMultiTensorsMultiPinnedWithConflict(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)]
        t_pre = [
            SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=tuple(shard_count*i + d for d in range(shard_count)), devices_pinned=(i < 2))
            for i in range(tensor_count)
        ]
        try:
            ops.transfer_if_needed(*t_pre)
        except ValueError:
            return

        assert False  # Should throw and error since the first two tensors are pinned to different devices


class MatmulTest(unittest.TestCase):
    def testShardedParallelAxesInLhsAndRhs(self):  # matmul_split
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count, devices=tuple(range(shard_count)), devices_pinned=True)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=1, shard_count=shard_count, devices=tuple(1 + i for i in range(shard_count)), devices_pinned=False)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        for i in range(shard_count):
            assert res_sharded.devices[i] == a_sharded.devices[i]  # A is pinned, result should be on its device
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)