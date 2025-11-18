# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import shutil
import tempfile
import unittest

import pytest
import torch

from iree.turbine.aot import ExternalTensorTrait
from amdsharktank.types import *
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.testing import assert_tensor_close

logger = get_logger(__name__)


def _t(name: str, *dims: int):
    return DefaultPrimitiveTensor(name=name, data=torch.ones(*dims))


def _flat_t_dict(*ts):
    return {t.name: t for t in ts}


class ThetaTest(unittest.TestCase):
    def testThetaAccess(self):
        # TODO: Make construction of a Theta programatically more natural.
        theta = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )

        # No root tensors.
        self.assertFalse(theta.tensors)

        flattened = theta.flatten()
        self.assertIn("a.b.c", flattened)
        self.assertIn("a.c.d", flattened)
        self.assertIn("1.2.3", flattened)

        logger.debug(theta.keys)
        self.assertIn("a", theta.keys)
        self.assertIn("1", theta.keys)

        sub_theta = theta("a")
        self.assertIn("b", sub_theta.keys)
        self.assertIn("c", sub_theta.keys)
        self.assertEqual("a.b.c", sub_theta.tensor("b", "c").name)
        self.assertEqual("a.c.d", sub_theta.tensor("c", "d").name)

        sub_sub_theta = theta("a", "b")
        self.assertEqual("a.b.c", sub_sub_theta.tensors[0].name)

    def testTransform(self):
        t1 = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )

        # We are mainly seeing that the structure/tensors were changed.
        # Without a second device, it is otherwise hard to see an effect.
        t2 = t1.to(device="cpu:1")
        self.assertIsNot(t1, t2)
        it1 = t1.tensor("a", "b", "c")
        it2 = t2.tensor("a", "b", "c")
        self.assertIsNot(it1, it2)
        for k in it1.subtensors.keys():
            pt1 = it1.subtensors[k]
            pt2 = it2.subtensors[k]
            self.assertIsNot(pt1, pt2)
            torch.testing.assert_close(pt1, pt2)

    def testPop(self):
        t1 = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("a.b.3", 3, 4),
            )
        )
        popped = t1.pop("a.b").flatten()
        t1 = t1.flatten()

        self.assertIsNotNone("a.c.d", t1.keys())
        self.assertNotIn("a.b.c", t1.keys())
        self.assertNotIn("a.b.3", t1.keys())
        self.assertIn("a.b.3", popped.keys())


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp("_dstest"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def testDatasetTransform(self):
        t1 = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )
        ds = Dataset({}, t1)
        ds.to(device="cpu:1")
        # Just checking that it was in fact transformed. Rely on other
        # unit tests for leaves transformed correctly.
        self.assertIsNot(t1, ds.root_theta)

    def testDatasetRoundtrip(self):
        theta = Theta(
            _flat_t_dict(
                _t("a.b.c", 1, 2),
                _t("a.c.d", 10, 11),
                _t("1.2.3", 3, 4),
            )
        )

        ds_src = Dataset({"foo": "bar"}, theta)
        ds_src.save(self.temp_dir / "myds.irpa")

        # Not mmap'ing makes things a bit better on Windows for tests.
        ds_load = Dataset.load(self.temp_dir / "myds.irpa", mmap=False)
        self.assertEqual("bar", ds_load.properties["foo"])
        t_abc = ds_load.root_theta.tensor("a", "b", "c")
        t_acd = ds_load.root_theta.tensor("a", "c", "d")
        t_123 = ds_load.root_theta.tensor("1", "2", "3")
        self.assertEqual([1, 2], list(t_abc.shape))
        self.assertEqual([10, 11], list(t_acd.shape))
        self.assertEqual([3, 4], list(t_123.shape))
        self.assertEqual(
            "a.b.c", ExternalTensorTrait.get(t_abc.as_torch()).external_name
        )
        self.assertEqual(
            "a.c.d", ExternalTensorTrait.get(t_acd.as_torch()).external_name
        )

    def testRoundTripDefaultPrimitiveTensor(self):
        t_orig = DefaultPrimitiveTensor(
            name="primitive", data=torch.Tensor([1.0, 2.0, 3.0])
        )
        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "p_ds.irpa")

        ds_load = Dataset.load(self.temp_dir / "p_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor(t_orig.name)
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_orig.shape)
        assert_tensor_close(t_load, t_orig, atol=0, rtol=0)

    def testRoundTripReplicatedTensor(self):
        shard_count = 4
        shards = [torch.Tensor([1.0, 2.0, 3.0])] * shard_count
        t_orig = ReplicatedTensor(name="rt", ts=shards)
        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "r_ds.irpa")

        ds_load = Dataset.load(self.temp_dir / "r_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor(t_orig.name)
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_orig.shape)
        self.assertEqual(t_load.shard_count, t_orig.shard_count)
        assert_tensor_close(t_load, t_orig, atol=0, rtol=0)

    def testRoundTripSplitTensor(self):
        shard_count = 4
        shards = [torch.Tensor([[1.0, 2.0, 3.0]])] * shard_count
        t_replicated = SplitPrimitiveTensor(name="st", ts=shards, shard_dim=1)
        ds_orig = Dataset({}, Theta({t_replicated.name: t_replicated}))
        ds_orig.save(self.temp_dir / "s_ds.irpa")

        ds_load = Dataset.load(self.temp_dir / "s_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor(t_replicated.name)
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_replicated.shape)
        self.assertEqual(t_load.shard_count, t_replicated.shard_count)
        assert_tensor_close(t_load, t_replicated, atol=0, rtol=0)

    def testRoundTripUnreducedTensor(self):
        shard_count = 4
        shards = [torch.Tensor([[1.0, 2.0, 3.0]])] * shard_count
        t_replicated = UnreducedTensor(name="t", ts=shards)
        ds_orig = Dataset({}, Theta({t_replicated.name: t_replicated}))
        ds_orig.save(self.temp_dir / "u_ds.irpa")

        ds_load = Dataset.load(self.temp_dir / "u_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor(t_replicated.name)
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_replicated.shape)
        self.assertEqual(t_load.shard_count, t_replicated.shard_count)
        assert_tensor_close(t_load, t_replicated, atol=0, rtol=0)

    def _createTestLayout(self):
        n = 128
        k = 1024
        bs = 32

        return BlockScaledLayout(
            [n, k],
            d=torch.rand(n, k // bs, 1, dtype=torch.float32),
            qs=torch.randint(0, 10, (n, k // bs, bs), dtype=torch.int8),
            m=torch.rand(n, k // bs, bs, dtype=torch.float32),
        )

    def testRoundtripPlanarQuantizedTensor(self):
        layout_in = self._createTestLayout()
        t_orig = PlanarQuantizedTensor(
            name="a.b.c", shape=layout_in.shape, layout=layout_in
        )
        self.assertIs(t_orig, t_orig.to_planar())
        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "myds.irpa")

        ds_load = Dataset.load(self.temp_dir / "myds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor("a", "b", "c")
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_orig.shape)
        self.assertEqual(t_load.subtensors.keys(), t_orig.subtensors.keys())
        self.assertEqual(t_load.layout.shape, t_orig.layout.shape)
        self.assertEqual(t_load.layout.planes.keys(), t_orig.layout.planes.keys())
        self.assertEqual(t_load.layout.metadata, t_orig.layout.metadata)

    @pytest.mark.skip(reason="Not implmented yet")
    def testRoundTripReplicatedPlanarQuantizedTensor(self):
        layout_in = self._createTestLayout()
        t_shard = PlanarQuantizedTensor(
            name="pqt", shape=layout_in.shape, layout=layout_in
        )
        self.assertIs(t_shard, t_shard.to_planar())
        t_orig = ReplicatedTensor(name="rpqt", ts=t_shard, shard_count=4)
        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "rq_ds.irpa")

        ds_load = Dataset.load(self.temp_dir / "rq_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor("rpqt")
        logger.debug(t_load)
        self.assertEqual(t_load.shape, t_orig.shape)
        self.assertEqual(t_load.shard_count, t_orig.shard_count)
        self.assertEqual(t_load.subtensors.keys(), t_orig.subtensors.keys())
        for i in range(t_load.shard_count):
            self.assertEqual(
                t_load.shards[i].layout.metadata, t_orig.shards[i].layout.metadata
            )
            for k in t_load.shards[i].subtensors.keys():
                assert_tensor_close(
                    t_load.shards[i].subtensors[k],
                    t_orig.shards[i].subtensors[k],
                    atol=0,
                    rtol=0,
                )

    @pytest.mark.skip(reason="Not implmented yet")
    def testRoundTripSplitPlanarQuantizedTensor(self):
        from copy import deepcopy

        layout_in = self._createTestLayout()
        t_shard = PlanarQuantizedTensor(
            name="pqt", shape=layout_in.shape, layout=layout_in
        )
        self.assertIs(t_shard, t_shard.to_planar())

        shards = [deepcopy(t_shard) for _ in range(4)]
        t_orig = SplitPrimitiveTensor(name="spqt", ts=shards, shard_dim=1)

        ds_orig = Dataset({}, Theta({t_orig.name: t_orig}))
        ds_orig.save(self.temp_dir / "sq_ds.irpa")
        ds_load = Dataset.load(self.temp_dir / "sq_ds.irpa", mmap=False)
        t_load = ds_load.root_theta.tensor("spqt")
        logger.debug(t_load)

        self.assertEqual(t_load.shape, t_orig.shape)
        self.assertEqual(t_load.shard_count, t_orig.shard_count)
        self.assertEqual(t_load.shard_dim, t_orig.shard_dim)
        self.assertEqual(t_load.subtensors.keys(), t_orig.subtensors.keys())
        for i in range(t_load.shard_count):
            self.assertEqual(
                t_load.shards[i].layout.metadata, t_orig.shards[i].layout.metadata
            )
            for k in t_load.shards[i].subtensors.keys():
                assert_tensor_close(
                    t_load.shards[i].subtensors[k],
                    t_orig.shards[i].subtensors[k],
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
