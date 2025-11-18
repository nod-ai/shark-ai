# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

import torch

from amdsharktank.ops.sharded_impls import assert_on_same_devices
from amdsharktank import ops
from amdsharktank.types import *
from amdsharktank.utils import iterables_equal
from amdsharktank.utils.testing import assert_tensor_close


class CheckThatOnSameDevicesTest(unittest.TestCase):
    def testOnSameDevices(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        ts_pre = [
            SplitPrimitiveTensor(
                shard_dim=1,
                ts=shards,
                devices=tuple(shard_count + d for d in range(shard_count)),
            )
            for _ in range(tensor_count)
        ]
        assert_on_same_devices(*ts_pre)

    def testOnDifferentDevices(self):
        tensor_count = 5
        shard_count = 4
        shard_shape = [3, 4]
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        t_pre = [
            SplitPrimitiveTensor(
                shard_dim=1,
                ts=shards,
                devices=tuple(shard_count * i + d for d in range(shard_count)),
            )
            for i in range(tensor_count)
        ]
        try:
            assert_on_same_devices(*t_pre)
        except ValueError:
            return

        assert False  # Should throw and error since the first two tensors are on different devices


class AllGatherTest(unittest.TestCase):
    def testAllGather(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        expected_result = torch.cat(shards, dim=shard_dim)

        devices = (0, 6, 1)
        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards, devices=devices)
        actual_result = ops.all_gather(sharded)

        for i in range(shard_count):
            assert_tensor_close(actual_result.shards[i], expected_result)
            assert actual_result.devices[i] == devices[i]


class AllReduceTest(unittest.TestCase):
    def testAllReduce(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        expected_result = torch.add(torch.add(shards[0], shards[1]), shards[2])

        devices = (0, 6, 1)
        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards, devices=devices)
        actual_result = ops.all_reduce(sharded)

        for i in range(shard_count):
            assert_tensor_close(actual_result.shards[i], expected_result)
            assert actual_result.devices[i] == devices[i]


class CatTest(unittest.TestCase):
    def testCatSplitDim(self):
        """Concatenation along the sharded split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 1
        # Done to ensure no overlap with default devices so incorrect placement is caught
        devices = tuple(range(shard_count, 2 * shard_count))

        a = torch.rand(3, 6, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim, devices=devices
        )

        sharded_a = ops.reshard_split(
            a, count=shard_count, dim=shard_dim, devices=devices
        )
        sharded_b = ops.reshard_split(
            b, count=shard_count, dim=shard_dim, devices=devices
        )

        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)
        assert iterables_equal(expected_result.devices, actual_result.devices)

    def testCatNonSplitDim(self):
        """Concatenation along a non-split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 0
        # Done to ensure no overlap with default devices so incorrect placement is caught
        devices = tuple(range(shard_count, 2 * shard_count))

        a = torch.rand(5, 4, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim, devices=devices
        )

        sharded_a = ops.reshard_split(
            a, count=shard_count, dim=shard_dim, devices=devices
        )
        sharded_b = ops.reshard_split(
            b, count=shard_count, dim=shard_dim, devices=devices
        )

        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)
        assert iterables_equal(expected_result.devices, actual_result.devices)

    def testCatReplicatedShouldFail(self):
        """Concatenation of replicated tensors."""
        shard_count = 2
        cat_dim = 0
        # Done to ensure no overlap with default devices so incorrect placement is caught
        devices = tuple(range(shard_count, 2 * shard_count))

        a = torch.rand(5, 4, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)

        sharded_a = ReplicatedTensor(ts=a, shard_count=shard_count, devices=devices)
        sharded_b = ReplicatedTensor(
            ts=b, shard_count=shard_count, devices=tuple(range(shard_count))
        )

        with pytest.raises(
            ValueError, match="All tensors must be placed on the same devices"
        ):
            ops.cat([sharded_a, sharded_b], dim=cat_dim)


class CloneTest(unittest.TestCase):
    def testCloneReplicatedFail(self):
        original = ReplicatedTensor(
            ts=torch.rand(5, 4, dtype=torch.float32), shard_count=4
        )
        try:
            original.clone(shards=None)
        except:
            return
        assert (
            False
        ), "Should have thrown an error when passing incorrect keywords to clone"

    def testCloneSplitFail(self):
        original = SplitPrimitiveTensor(
            ts=torch.rand(5, 4, dtype=torch.float32), shard_dim=1, shard_count=4
        )
        try:
            original.clone(shards=None)
        except:
            return
        assert (
            False
        ), "Should have thrown an error when passing incorrect keywords to clone"

    def testCloneUnreducedFail(self):
        original = UnreducedTensor(ts=[torch.rand(5, 4, dtype=torch.float32)])
        try:
            original.clone(shards=None)
        except:
            return
        assert (
            False
        ), "Should have thrown an error when passing incorrect keywords to clone"


class IndexSelectTest(unittest.TestCase):
    def testIndexReplicatedPinned(self):
        shard_count = 5
        shards = [torch.rand(5, 4, dtype=torch.float32) for _ in range(shard_count)]
        devices = tuple(5 + i for i in range(shard_count))
        base = ReplicatedTensor(ts=shards, devices=devices)
        indices = torch.tensor([0, 3, 1, 4], dtype=torch.int64)
        indices_t = ReplicatedTensor(
            ts=indices, shard_count=shard_count, devices=devices
        )

        expected_results = [torch.index_select(shard, 0, indices) for shard in shards]

        actual_result = ops.index_select(base, 0, indices_t)
        assert iterables_equal(devices, actual_result.devices)
        for expected_shard, actual_shards in zip(
            expected_results, actual_result.shards
        ):
            assert expected_shard.equal(actual_shards.as_torch())


class MatmulTest(unittest.TestCase):
    def testShardedParallelAxesInLhsAndRhs(self):  # matmul_split
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)

        shard_count = 3
        devices = tuple(1 + 2 * i for i in range(shard_count))
        a_sharded = SplitPrimitiveTensor(
            ts=a,
            shard_dim=1,
            shard_count=shard_count,
            devices=devices,
        )
        b_sharded = SplitPrimitiveTensor(
            ts=b,
            shard_dim=1,
            shard_count=shard_count,
            devices=devices,
        )
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        for i in range(shard_count):
            assert devices[i] == res_sharded.devices[i]
        actual_result = ops.sharded_cat(res_sharded)
        assert_tensor_close(actual_result, expected_result)


class TransposeTest(unittest.TestCase):
    def testTranspose(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2 + i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices)

        post = pre.T
        assert iterables_equal(pre.devices, post.devices)
