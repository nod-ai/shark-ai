# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable
import unittest
import torch

from parameterized import parameterized

from sharktank import ops
from sharktank.ops.sharded_impls import zeros_replicated
from sharktank.types import DefaultPrimitiveTensor
from sharktank.ops.default_impls import abs_default, cos_default, zeros_default
from sharktank.types.tensors import InferenceTensor, ReplicatedTensor
from sharktank.utils.testing import assert_tensor_close


class DispatchTest(unittest.TestCase):
    def setUp(self):
        ops._registry._test_enable_last_op_dispatch(True)

    def tearDown(self):
        ops._registry._test_enable_last_op_dispatch(False)

    def make_tensor(self, shards: int) -> InferenceTensor:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        if shards == 1:
            return DefaultPrimitiveTensor(data=tensor)
        else:
            return ReplicatedTensor(ts=tensor, shard_count=shards)

    @parameterized.expand(
        [
            (None, zeros_default),
            ((2, 3), zeros_replicated._unwrapped),
        ]
    )
    def test_non_trivially_replicable_op(
        self, devices: tuple[int, ...] | None, expected_dispatch: Callable
    ):
        ops.zeros(1, devices=devices)

        last_dispatch_after_zeros = ops._registry._test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_zeros, expected_dispatch)

    @parameterized.expand([(1,), (2,)])
    def test_trivially_replicable_op(self, shard_count: int):
        self.make_tensor(shard_count).abs()

        last_dispatch = ops._registry._test_get_last_op_dispatch()
        self.assertIs(last_dispatch, abs_default)

    @parameterized.expand([(1,), (2,)])
    def test_multiple_dispatch(self, shard_count: int):
        tensor = self.make_tensor(shard_count)

        tensor.abs()
        last_dispatch_after_abs = ops._registry._test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_abs, abs_default)

        tensor.cos()
        last_dispatch_after_cos = ops._registry._test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_cos, cos_default)


if __name__ == "__main__":
    unittest.main()
