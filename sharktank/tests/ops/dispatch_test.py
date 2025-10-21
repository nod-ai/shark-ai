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
from sharktank.types import DefaultPrimitiveTensor, InferenceTensor, ReplicatedTensor
from sharktank.ops.sharded_impls import zeros_replicated, transfer_n_pin
from sharktank.ops.default_impls import abs_default, cos_default, zeros_default
from sharktank.ops._registry import (
    unwrap_if_possible,
    _test_enable_last_op_dispatch,
    _test_get_last_op_dispatch,
)
from sharktank.ops.utils import trivially_replicable


class DispatchTest(unittest.TestCase):
    def setUp(self):
        _test_enable_last_op_dispatch(True)

    def tearDown(self):
        _test_enable_last_op_dispatch(False)

    def make_tensor(self, shards: int) -> InferenceTensor:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        if shards == 1:
            return DefaultPrimitiveTensor(data=tensor)
        else:
            return ReplicatedTensor(ts=tensor, shard_count=shards)

    @parameterized.expand(
        [
            (None, unwrap_if_possible(zeros_default)),
            ((2, 3), unwrap_if_possible(zeros_replicated)),
        ]
    )
    def test_non_trivially_replicable_op(
        self, devices: tuple[int, ...] | None, expected_dispatch: Callable
    ):
        ops.zeros(1, devices=devices)

        last_dispatch_after_zeros = _test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_zeros, expected_dispatch)

    @parameterized.expand([(1,), (2,)])
    def test_trivially_replicable_op(self, shard_count: int):
        self.make_tensor(shard_count).abs()

        last_dispatch = _test_get_last_op_dispatch()
        self.assertIs(last_dispatch, unwrap_if_possible(abs_default))

    @parameterized.expand([(1,), (2,)])
    def test_multiple_dispatch(self, shard_count: int):
        tensor = self.make_tensor(shard_count)

        tensor.abs()
        last_dispatch_after_abs = _test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_abs, unwrap_if_possible(abs_default))

        tensor.cos()
        last_dispatch_after_cos = _test_get_last_op_dispatch()
        self.assertIs(last_dispatch_after_cos, unwrap_if_possible(cos_default))


def f(*args, **kwargs) -> torch.Tensor:
    ...


class UnwrapIfPossibleTest(unittest.TestCase):
    def test_unwrap_no_wrapper(self):
        self.assertIs(unwrap_if_possible(f), f)

    @parameterized.expand([transfer_n_pin])
    def test_unwrap_with_wrapper(self, wrapping_fn: Callable):
        f_wrapped = wrapping_fn(f)
        self.assertIsNot(f_wrapped, f)
        self.assertIs(unwrap_if_possible(f_wrapped), f)

    def test_unwrap_with_trivially_replicable_wrapper(self):
        f_wrapped = trivially_replicable(f)
        self.assertIsNot(unwrap_if_possible(f_wrapped), f)
        assert f_wrapped._trivially_replicable_wrapper


if __name__ == "__main__":
    unittest.main()
