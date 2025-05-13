# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from sharktank import ops
from sharktank.types import PlanarQuantizedTensor, TensorScaledLayout


class elementwise_tensor_tensor_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, 1e-2, 1e-3),
            (torch.float16, 1e-2, 1e-3),
        ]
    )
    def test_mul_basic(self, dtype, atol, rtol):
        shape = [17, 32]
        x = torch.rand(shape, dtype=dtype)
        y = torch.rand(shape, dtype=dtype)
        one = torch.tensor(1.0).to(dtype=dtype)
        x_layout = TensorScaledLayout(shape=shape, qs=x, d=one)
        x_qtensor = PlanarQuantizedTensor(shape=shape, layout=x_layout)
        y_layout = TensorScaledLayout(shape=shape, qs=y, d=one)
        y_qtensor = PlanarQuantizedTensor(shape=shape, layout=y_layout)
        result = ops.elementwise(torch.mul, x_qtensor, y_qtensor)

        ref = x * y
        torch.testing.assert_close(result.unpack().dequant(), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float32, 1e-2, 1e-3),
            (torch.float16, 1e-2, 1e-3),
        ]
    )
    def test_mul_broadcast(self, dtype, atol, rtol):
        shapex = [2, 3, 1]
        shapey = [1, 3, 5]
        x = torch.rand(shapex, dtype=dtype)
        y = torch.rand(shapey, dtype=dtype)
        one = torch.tensor(1.0).to(dtype=dtype)
        x_layout = TensorScaledLayout(shape=shapex, qs=x, d=one)
        x_qtensor = PlanarQuantizedTensor(shape=shapex, layout=x_layout)
        y_layout = TensorScaledLayout(shape=shapey, qs=y, d=one)
        y_qtensor = PlanarQuantizedTensor(shape=shapey, layout=y_layout)
        result = ops.elementwise(torch.mul, x_qtensor, y_qtensor)

        ref = x * y
        print(ref)
        print(result.unpack().dequant())
        torch.testing.assert_close(result.unpack().dequant(), ref, atol=atol, rtol=rtol)

class elementwise_tensor_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (torch.float32, 1e-2, 1e-3),
            (torch.float16, 1e-2, 1e-3),
        ]
    )

    def test_square(self, dtype, atol, rtol):
        shape = [3, 7, 17]
        x = torch.rand(shape, dtype=dtype)
        one = torch.tensor(1.0).to(dtype)
        x_layout = TensorScaledLayout(shape=shape, qs=x, d=one)
        x_qtensor = PlanarQuantizedTensor(shape=shape, layout=x_layout)
        result = ops.elementwise(torch.square, x_qtensor)

        ref = torch.square(x)
        torch.testing.assert_close(result.unpack().dequant(), ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
