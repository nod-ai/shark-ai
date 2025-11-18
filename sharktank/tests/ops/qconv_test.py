# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
import torch.nn.functional as F

from amdsharktank import ops
from amdsharktank.types import *


def _randomize_per_axis(t: torch.Tensor, axis: int, offset_range: float = 0.0):
    # Applies a randomized per-axis scale and offset to a tensor.
    bcast_shape = [1] * len(t.shape)
    bcast_shape[axis] = t.shape[axis]

    rnd_mult = torch.rand(bcast_shape, dtype=torch.float32)
    t = t * rnd_mult
    rnd_offset = torch.rand(bcast_shape, dtype=torch.float32) * offset_range
    return t + rnd_offset


def _scale_offset_per_axis_i8(t: torch.Tensor, reduce_dim: int):
    mn, _ = torch.min(t, reduce_dim)
    mx, _ = torch.max(t, reduce_dim)
    scale = 255.0 / (mx - mn)
    offset = torch.round(mn * scale) - 128
    return scale, offset.to(dtype=torch.int8)


def _scale_per_tensor_i8(t: torch.Tensor):
    amax = torch.abs(torch.max(t))
    scale = 127 / amax.clamp(1e-6)
    return scale


class QConvTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def tearDown(self):
        ops._registry._test_enable_last_op_dispatch(False)

    def testInputSymPerTensor_WeightAsymPerChannel_NoBias(self):
        ops._registry._test_enable_last_op_dispatch(True)
        input = _randomize_per_axis(
            torch.rand(4, 8, 16, 16, dtype=torch.float32), axis=1
        )
        weight = _randomize_per_axis(
            torch.rand(8, 8, 4, 4, dtype=torch.float32), axis=0, offset_range=0.2
        )

        input_scale = _scale_per_tensor_i8(input)
        weight_scale, weight_offset = _scale_offset_per_axis_i8(weight.flatten(1), 1)

        input_q = StaticScaledQuantizer(scale=input_scale, dtype=torch.int8).quantize(
            input
        )
        weight_q = StaticScaledQuantizer(
            scale=weight_scale, offset=weight_offset, dtype=torch.int8, axis=0
        ).quantize(weight)

        y_quantized = ops.conv2d(input_q, weight_q, bias=None, stride=1, padding=(1, 1))
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.qconv_impls.qconv2d_tensor_scaled,
        )
        y_actual = y_quantized.unpack().dequant()

        y_ref = torch.nn.functional.conv2d(
            input_q.unpack().dequant(),
            weight_q.unpack().dequant(),
            bias=None,
            stride=1,
            padding=(1, 1),
        )
        torch.testing.assert_close(y_actual, y_ref)

    def testInputSymPerTensor_WeightAsymPerChannel_FloatBias(self):
        ops._registry._test_enable_last_op_dispatch(True)
        input = _randomize_per_axis(
            torch.rand(4, 8, 16, 16, dtype=torch.float32), axis=1
        )
        weight = _randomize_per_axis(
            torch.rand(8, 8, 4, 4, dtype=torch.float32), axis=0, offset_range=0.2
        )
        bias = torch.rand(8, dtype=torch.float32) + 5.0

        input_scale = _scale_per_tensor_i8(input)
        weight_scale, weight_offset = _scale_offset_per_axis_i8(weight.flatten(1), 1)

        input_q = StaticScaledQuantizer(scale=input_scale, dtype=torch.int8).quantize(
            input
        )
        weight_q = StaticScaledQuantizer(
            scale=weight_scale, offset=weight_offset, dtype=torch.int8, axis=0
        ).quantize(weight)

        y_actual = ops.conv2d(input_q, weight_q, bias, stride=1, padding=(1, 1))
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.qconv_impls.qconv2d_tensor_scaled,
        )
        y_ref = torch.nn.functional.conv2d(
            input_q.unpack().dequant(),
            weight_q.unpack().dequant(),
            bias,
            stride=1,
            padding=(1, 1),
        )
        torch.testing.assert_close(y_actual, y_ref)

    def testInputSymPerTensor_WeightAsymPerChannel_QuantizedBias(self):
        ops._registry._test_enable_last_op_dispatch(True)
        input = _randomize_per_axis(
            torch.rand(4, 8, 16, 16, dtype=torch.float32), axis=1
        )
        weight = _randomize_per_axis(
            torch.rand(8, 8, 4, 4, dtype=torch.float32), axis=0, offset_range=0.2
        )
        bias = torch.rand(8, dtype=torch.float32) + 5.0

        input_scale = _scale_per_tensor_i8(input)
        weight_scale, weight_offset = _scale_offset_per_axis_i8(weight.flatten(1), 1)
        bias_scale = input_scale * weight_scale

        input_q = StaticScaledQuantizer(scale=input_scale, dtype=torch.int8).quantize(
            input
        )
        weight_q = StaticScaledQuantizer(
            scale=weight_scale, offset=weight_offset, dtype=torch.int8, axis=0
        ).quantize(weight)
        bias_q = StaticScaledQuantizer(
            scale=bias_scale, dtype=torch.int32, axis=0
        ).quantize(bias)

        y_quantized = ops.conv2d(input_q, weight_q, bias_q, stride=1, padding=(1, 1))
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.qconv_impls.qconv2d_tensor_scaled,
        )
        y_actual = y_quantized.unpack().dequant()
        y_ref = torch.nn.functional.conv2d(
            input_q.unpack().dequant(),
            weight_q.unpack().dequant(),
            bias_q.unpack().dequant(),
            stride=1,
            padding=(1, 1),
        )
        torch.testing.assert_close(y_actual, y_ref)

    def testInputSymPerTensor_WeightSymPerTensor_NoBias(self):
        ops._registry._test_enable_last_op_dispatch(True)
        input = _randomize_per_axis(
            torch.rand(4, 8, 16, 16, dtype=torch.float32), axis=1
        )
        weight = _randomize_per_axis(
            torch.rand(8, 8, 4, 4, dtype=torch.float32), axis=0, offset_range=0.2
        )

        input_scale = _scale_per_tensor_i8(input)
        weight_scale = _scale_per_tensor_i8(weight)

        input_q = StaticScaledQuantizer(scale=input_scale, dtype=torch.int8).quantize(
            input
        )
        weight_q = StaticScaledQuantizer(scale=weight_scale, dtype=torch.int8).quantize(
            weight
        )

        y_quantized = ops.conv2d(input_q, weight_q, bias=None, stride=1, padding=(1, 1))
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.qconv_impls.qconv2d_tensor_scaled,
        )
        y_actual = y_quantized.unpack().dequant()
        y_ref = torch.nn.functional.conv2d(
            input_q.unpack().dequant(),
            weight_q.unpack().dequant(),
            bias=None,
            stride=1,
            padding=(1, 1),
        )
        torch.testing.assert_close(y_actual, y_ref)

    @unittest.skip("Bug in joint offset application #55")
    def testInputAsymPerChannel_WeightAsymPerChannel_NoBias(self):
        ops._registry._test_enable_last_op_dispatch(True)
        input = _randomize_per_axis(
            torch.rand(4, 8, 16, 16, dtype=torch.float32), axis=1, offset_range=-0.2
        )
        weight = _randomize_per_axis(
            torch.rand(8, 8, 4, 4, dtype=torch.float32), axis=0, offset_range=0.2
        )

        input_scale, input_offset = _scale_offset_per_axis_i8(
            input.transpose(0, 1).flatten(1), 1
        )
        weight_scale, weight_offset = _scale_offset_per_axis_i8(weight.flatten(1), 1)

        input_q = StaticScaledQuantizer(
            scale=input_scale, offset=input_offset, dtype=torch.int8, axis=1
        ).quantize(input)
        weight_q = StaticScaledQuantizer(
            scale=weight_scale, offset=weight_offset, dtype=torch.int8, axis=0
        ).quantize(weight)

        y_actual = (
            ops.conv2d(input_q, weight_q, bias=None, stride=1, padding=(1, 1))
            .unpack()
            .dequant()
        )
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.qconv_impls.qconv2d_tensor_scaled,
        )
        y_ref = torch.nn.functional.conv2d(
            input_q.unpack().dequant(),
            weight_q.unpack().dequant(),
            bias=None,
            stride=1,
            padding=(1, 1),
        )
        torch.testing.assert_close(y_actual, y_ref)


if __name__ == "__main__":
    unittest.main()
