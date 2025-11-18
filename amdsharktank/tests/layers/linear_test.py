# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import torch
from itertools import product
from parameterized import parameterized

from amdsharktank.layers import *
from amdsharktank.types import *
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.random import make_rand_torch
from amdsharktank.types.pipelining import parallelize_in_place
import amdsharktank.ops as ops

logger = get_logger(__name__)


def _randomize_per_axis(t: torch.Tensor, axis: int, offset_range: float = 0.0):
    # Applies a randomized per-axis scale and offset to a tensor.
    bcast_shape = [1] * len(t.shape)
    bcast_shape[axis] = t.shape[axis]

    rnd_mult = torch.rand(bcast_shape, dtype=torch.float32)
    t = t * rnd_mult
    rnd_offset = torch.rand(bcast_shape, dtype=torch.float32) * offset_range
    return t + rnd_offset


def _scale_offset_per_axis_ui8(t: torch.Tensor, reduce_dim: int):
    mn, _ = torch.min(t, reduce_dim)
    mx, _ = torch.max(t, reduce_dim)
    scale = 255.0 / (mx - mn)
    offset = torch.round(mn * scale)
    return scale, offset.to(dtype=torch.uint8)


def _scale_per_tensor_i8(t: torch.Tensor):
    amax = torch.abs(torch.max(t))
    scale = 127 / amax.clamp(1e-6)
    return scale


_cases = [
    (torch.bfloat16, torch.float32, torch.float8_e4m3fnuz, False, False, 1e-2),
    (torch.bfloat16, torch.float32, torch.float8_e4m3fnuz, False, True, 1e-2),
    (torch.float32, torch.float32, torch.float8_e4m3fnuz, False, False, 1e-6),
    (torch.float32, torch.float32, torch.float8_e4m3fnuz, False, True, 1e-6),
    (torch.float32, torch.float32, torch.float16, True, False, 1e-6),
    (torch.float32, torch.float32, torch.float16, False, False, 1e-6),
    (torch.float32, torch.float32, torch.float16, False, True, 1e-6),
    (torch.float32, torch.float32, torch.float32, True, False, 1e-6),
]


class LinearQuantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def testNativeQuant_SymPerTensor_AsymPerAxis0_Dynamic(self):
        # Tests a linear layer that multiplies a per-tensor lhs with a
        # per-axis(0) rhs to produce a dynamically scaled FP result as output.

        # Generate random tensors that are then randomly scaled along axis-0.
        # Bias the rhs slightly to induce more interesting zero points.
        lhs = _randomize_per_axis(torch.rand(4, 8, 128, dtype=torch.float32), axis=0)
        rhs = _randomize_per_axis(
            torch.rand(16, 128, dtype=torch.float32), axis=0, offset_range=0.02
        )
        bias = torch.rand(16, dtype=torch.float32) + 5.0
        # bias = torch.zeros(16, dtype=torch.float32)

        lhs_scale = _scale_per_tensor_i8(lhs)
        rhs_scale, rhs_offset = _scale_offset_per_axis_ui8(rhs, 1)
        bias_scale = lhs_scale * rhs_scale

        lhs_quantizer = StaticScaledQuantizer(
            name="q_input", scale=lhs_scale, dtype=torch.int8
        )
        rhs_quantizer = StaticScaledQuantizer(
            scale=rhs_scale, offset=rhs_offset, dtype=torch.uint8, axis=0
        )
        rhs_quant = rhs_quantizer.quantize(rhs, name="weight")
        bias_quantizer = StaticScaledQuantizer(
            scale=bias_scale, dtype=torch.int32, axis=0
        )
        bias_quant = bias_quantizer.quantize(bias, name="bias")

        # Sanity check that dequant'ing the RHS is roughly the same.
        # rhs_dequant = rhs_quant.unpack().dequant()
        # print("RHS_DIFF:", torch.abs(rhs_dequant - rhs))
        # print("RHS:", rhs)
        # print("RHS_DEQUANT:", rhs_dequant)
        # torch.testing.assert_close(rhs_dequant, rhs, atol=1e-1, rtol=1e-2)

        theta = Theta(
            [
                lhs_quantizer,
                rhs_quant,
                bias_quant,
            ]
        )
        linear = LinearLayer(theta, fake_quant=False)

        output = linear(lhs)
        output_ref = torch.matmul(lhs, rhs.T) + bias
        print(torch.abs(output - output_ref))
        torch.testing.assert_close(output, output_ref, atol=1e-1, rtol=1e-1)

    # Paramaterize with product to create 2 variations of each entry with a True or
    @parameterized.expand(
        [(*case, shard_count) for case, shard_count in product(_cases, [None, 1, 2])],
    )
    def testPerTensorScale(
        self,
        dequantized_dtype: torch.dtype,
        quantized_scale_dtype: torch.dtype,
        quantized_dtype: torch.dtype,
        with_bias: bool,
        fake_quant: bool,
        atol: float,
        shard_count: int | None,
    ):
        """Test a linear layer where each tensor being quantized with a single
        different scale."""

        ref_dtype = torch.float64

        x = make_rand_torch([10, 8, 8], dtype=dequantized_dtype)
        input_scale = torch.tensor(0.5, dtype=quantized_scale_dtype)
        input_quantizer = StaticScaledQuantizer(
            name="q_input", scale=input_scale, dtype=quantized_dtype
        )
        # We roundtrip through quantization to know that any discrepancies in the
        # results come from the quantized linear operation itself. Not form the
        # inaccuracies of the initial quantization.
        x_dequantized = input_quantizer.quantize(x).unpack().dequant()
        torch.testing.assert_close(
            input_quantizer.quantize(x_dequantized).unpack().dequant(),
            x_dequantized,
            atol=0,
            rtol=0,
        )

        weight = make_rand_torch([12, x.shape[2]], dtype=dequantized_dtype)
        weight_scale = torch.tensor(0.66, dtype=quantized_scale_dtype)
        weight_quantizer = StaticScaledQuantizer(
            scale=weight_scale, dtype=quantized_dtype
        )
        weight_dequantized = weight_quantizer.quantize(weight).unpack().dequant()
        weight_quantized = weight_quantizer.quantize(weight_dequantized, name="weight")
        torch.testing.assert_close(
            weight_quantizer.quantize(weight_dequantized).unpack().dequant(),
            weight_dequantized,
            atol=0,
            rtol=0,
        )

        if with_bias:
            bias = make_rand_torch(
                [x.shape[1], weight.shape[0]], dtype=dequantized_dtype
            )
            bias_scale = torch.tensor(1.25, dtype=quantized_scale_dtype)
            bias_quantizer = StaticScaledQuantizer(
                scale=bias_scale, dtype=quantized_dtype
            )
            bias_dequantized = bias_quantizer.quantize(bias).unpack().dequant()
            bias_quantized = bias_quantizer.quantize(bias_dequantized, name="bias")
            torch.testing.assert_close(
                bias_quantizer.quantize(bias_dequantized).unpack().dequant(),
                bias_dequantized,
                atol=0,
                rtol=0,
            )

        expected = torch.matmul(
            x_dequantized.to(ref_dtype), weight_dequantized.T.to(ref_dtype)
        )
        if with_bias:
            expected += bias_dequantized.to(ref_dtype)

        if shard_count:
            input_quantizer = ReplicatedTensor(
                ts=input_quantizer, shard_count=shard_count, name="q_input"
            )
            weight_quantized = ReplicatedTensor(
                ts=weight_quantized, shard_count=shard_count, name="weight"
            )
            if with_bias:
                bias_quantized = ReplicatedTensor(
                    ts=bias_quantized, shard_count=shard_count, name="bias"
                )

            x_dequantized = ReplicatedTensor(ts=x_dequantized, shard_count=shard_count)

        theta_tensors = [
            input_quantizer,
            weight_quantized,
        ]
        if with_bias:
            theta_tensors += [bias_quantized]
        theta = Theta(theta_tensors)
        linear = LinearLayer(theta, fake_quant=fake_quant)
        actual = linear(x_dequantized)
        actual = ops.unbox_tensor(actual).to(dtype=expected.dtype)

        abs_diff = (expected - actual).abs()
        logger.info(
            f"abs diff from expected (std, mean, median) = {[float(abs_diff.std()), float(abs_diff.mean()), float(abs_diff.median())]}"
        )

        torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


if __name__ == "__main__":
    unittest.main()
