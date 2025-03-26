# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

import torch

from sharktank.ops.sharded_impls import transfer_if_needed
from sharktank import ops
from sharktank.types import *
from sharktank.utils import iterables_equal


class TransferIfNeededTest(unittest.TestCase):
    def testTransferOnSameDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2 + i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre_1 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices, pinned=True
        )
        pre_2 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices, pinned=False
        )

        post_1, post_2 = transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testTransferOnDifferentDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices_pinned = tuple(2 + i for i in range(shard_count))
        devices_free = tuple(2 + 2 * i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre_1 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices_pinned, pinned=True
        )
        pre_2 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices_free, pinned=False
        )

        post_1, post_2 = transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices_pinned):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testBothPinnedOnSameDevice(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2 + i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre_1 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices, pinned=True
        )
        pre_2 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices, pinned=True
        )

        post_1, post_2 = transfer_if_needed(pre_1, pre_2)
        for i, device in enumerate(devices):
            assert device == post_1.devices[i]
            assert device == post_2.devices[i]

    def testBothPinnedOnDifferentDevices(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices_pinned = tuple(2 + i for i in range(shard_count))
        devices_free = tuple(2 + 2 * i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre_1 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices_pinned, pinned=True
        )
        pre_2 = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices_free, pinned=True
        )

        try:
            transfer_if_needed(pre_1, pre_2)
        except ValueError:
            return
        assert False  # Should have thrown a ValueError since both tensors are pinned, but devices are not the same

    def testMultiTensorsNoPinnedSameDevice(self):
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
                pinned=False,
            )
            for _ in range(tensor_count)
        ]
        ts_post = transfer_if_needed(*ts_pre)

        for t_pre, t_post in zip(ts_pre, ts_post):
            assert iterables_equal(t_pre.devices, t_post.devices)
            assert t_pre.pinned == t_post.pinned

    def testMultiTensorsNoPinnedMultiDevice(self):
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
                pinned=False,
            )
            for i in range(tensor_count)
        ]

        try:
            transfer_if_needed(*t_pre)
        except ValueError:
            return
        assert False  # Should have thrown a ValueError since no devices are different but none are pinned

    def testMultiTensorsOnePinned(self):
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
                devices=tuple(shard_count * i + d for d in range(shard_count)),
                pinned=(i == 0),
            )
            for i in range(tensor_count)
        ]
        ts_post = transfer_if_needed(*ts_pre)

        for t_post in ts_post:
            assert iterables_equal(ts_pre[0].devices, t_post.devices)
            t_post.pinned == True

    def testMultiTensorsMultiPinnedNoConflict(self):
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
                devices=tuple(
                    shard_count * i * (i % 2 != 0) + d for d in range(shard_count)
                ),
                pinned=(i % 2 == 0),
            )
            for i in range(tensor_count)
        ]
        ts_post = transfer_if_needed(*ts_pre)

        for t_post in ts_post:
            assert iterables_equal(ts_pre[0].devices, t_post.devices)
            t_post.pinned == True

    def testMultiTensorsMultiPinnedWithConflict(self):
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
                pinned=(i < 2),
            )
            for i in range(tensor_count)
        ]
        try:
            transfer_if_needed(*t_pre)
        except ValueError:
            return

        assert False  # Should throw and error since the first two tensors are pinned to different devices


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
            torch.testing.assert_close(
                actual_result.shards[i].as_torch(), expected_result
            )
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
            torch.testing.assert_close(
                actual_result.shards[i].as_torch(), expected_result
            )
            assert actual_result.devices[i] == devices[i]


class CatTest(unittest.TestCase):
    def testCatSplitDimPinned(self):
        """Concatenation along the sharded split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 1
        a = torch.rand(3, 6, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim
        )
        expected_result = expected_result.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )

        sharded_a = ops.reshard_split(a, count=shard_count, dim=shard_dim)
        sharded_a = sharded_a.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )

        sharded_b = ops.reshard_split(b, count=shard_count, dim=shard_dim)
        sharded_b = sharded_b.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )
        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)
        assert iterables_equal(expected_result.devices, actual_result.devices)
        assert expected_result.pinned == actual_result.pinned

    def testCatNonSplitDimPinned(self):
        """Concatenation along a non-split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 0
        a = torch.rand(5, 4, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim
        )
        expected_result = expected_result.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )

        sharded_a = ops.reshard_split(a, count=shard_count, dim=shard_dim)
        sharded_a = sharded_a.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )

        sharded_b = ops.reshard_split(b, count=shard_count, dim=shard_dim)
        sharded_b = sharded_b.clone(
            devices=tuple(4 + i for i in range(shard_count)), pinned=True
        )
        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)
        assert iterables_equal(expected_result.devices, actual_result.devices)
        assert expected_result.pinned == actual_result.pinned


class IndexSelectTest(unittest.TestCase):
    def testIndexReplicatedPinned(self):
        shard_count = 5
        shards = [torch.rand(5, 4, dtype=torch.float32) for _ in range(shard_count)]
        devices = tuple(5 + i for i in range(shard_count))
        base = ReplicatedTensor(ts=shards, devices=devices, pinned=True)
        indices = torch.tensor([0, 3, 1, 4], dtype=torch.int64)
        # TODO: Manually overriding pinned=False may not reflect real usage when running with a parallelized model
        indices_t = ReplicatedTensor(ts=indices, shard_count=shard_count, pinned=False)

        expected_results = [torch.index_select(shard, 0, indices) for shard in shards]

        actual_result = ops.index_select(base, 0, indices_t)
        assert iterables_equal(devices, actual_result.devices)
        # assert actual_result.pinned  # TODO: Should these be pinned?
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
        a_sharded = SplitPrimitiveTensor(
            ts=a,
            shard_dim=1,
            shard_count=shard_count,
            devices=tuple(range(shard_count)),
            pinned=True,
        )
        b_sharded = SplitPrimitiveTensor(
            ts=b,
            shard_dim=1,
            shard_count=shard_count,
            devices=tuple(1 + i for i in range(shard_count)),
            pinned=False,
        )
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        for i in range(shard_count):
            assert (
                res_sharded.devices[i] == a_sharded.devices[i]
            )  # A is pinned, result should be on its device
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)


class ShardLikeTest(unittest.TestCase):
    def testReshardLikeUnshardedToReplicated(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        expected_result = ops.replicate(tensor, count=shard_count).clone(
            devices=tuple(2 * i for i in range(shard_count)), pinned=True
        )

        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)
        assert actual_result.pinned == expected_result.pinned
        assert iterables_equal(actual_result.devices, expected_result.devices)

    def testReshardLikeUnshardedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(
            tensor, dim=shard_dim, count=shard_count
        ).clone(devices=tuple(2 * i for i in range(shard_count)), pinned=True)

        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)
        assert actual_result.pinned == expected_result.pinned
        assert iterables_equal(actual_result.devices, expected_result.devices)

    def testReshardLikeShardedToShared(self):
        tensor = torch.rand(5, 6, dtype=torch.float32)
        shard_dim = 1
        shard_count = 3
        expected_result = ops.reshard_split(
            tensor, dim=shard_dim, count=shard_count
        ).clone(pinned=False)
        target = ops.reshard_split(tensor, dim=shard_dim, count=shard_count).clone(
            devices=tuple(2 * i for i in range(shard_count)), pinned=True
        )

        actual_result = ops.reshard_like(expected_result, target)
        assert expected_result.is_deep_equal(actual_result)
        assert actual_result.pinned == target.pinned
        assert iterables_equal(actual_result.devices, target.devices)

    def testReshardLikeReplicatedToReplicated(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_count = 2
        input_tensor = ops.replicate(tensor, count=shard_count).clone(
            devices=tuple(range(shard_count)), pinned=False
        )
        target = ops.replicate(tensor, count=shard_count).clone(
            devices=tuple(2 * i for i in range(shard_count)), pinned=True
        )

        actual_result = ops.reshard_like(input_tensor, target)
        assert input_tensor.is_deep_equal(actual_result)
        assert actual_result.pinned == target.pinned
        assert iterables_equal(actual_result.devices, target.devices)

    def testReshardLikeReplicatedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(
            tensor, dim=shard_dim, count=shard_count
        ).clone(devices=tuple(2 * i for i in range(shard_count)), pinned=True)
        replicated_tensor = ops.replicate(tensor, count=shard_count).clone(
            devices=tuple(range(shard_count)), pinned=False
        )

        actual_result = ops.reshard_like(replicated_tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)
        assert actual_result.pinned == expected_result.pinned
        assert iterables_equal(actual_result.devices, expected_result.devices)


class TransposeTest(unittest.TestCase):
    def testUnpinnedTranspose(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2 + i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre = SplitPrimitiveTensor(
            shard_dim=1, ts=shards, devices=devices, pinned=False
        )

        post = pre.T
        assert iterables_equal(pre.devices, post.devices)
        # NOTE: Can't compare .pinned. post gets pinned since resulting ShardedTensor is made with torch.Tensor shards which are assumed to always be pinned.
        # assert post.pinned == pre.pinned

    def testPinnedTranspose(self):
        shard_count = 4
        shard_shape = [3, 4]
        devices = tuple(2 + i for i in range(shard_count))
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for _ in range(shard_count)
        ]
        pre = SplitPrimitiveTensor(shard_dim=1, ts=shards, devices=devices, pinned=True)

        post = pre.T
        assert iterables_equal(pre.devices, post.devices)
        assert post.pinned == pre.pinned
