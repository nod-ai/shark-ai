# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
from .. import ops
from .base import Theta, ThetaLayer
from ..types import (
    DynamicScaledQuantizer,
    QuantizedTensor,
    QuantizerTensor,
    SplitPrimitiveTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
    PlanarQuantizedTensor,
)

__all__ = [
    "LinearLayer",
]


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    if premul_input is not None:
      x = x * premul_input
    matmul(x, weight.T) + bias

    fake quant only exists in order to allow for q_input to act as qdq.
    when fake quant is false, q_input will quantize normally.
    ```
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        bias_name: str = "bias",
        fake_quant: bool = False,
        experimental_mm_cache_size: Optional[int] = None,
        experimental_mm_cache_sets: Optional[int] = None,
    ):
        super().__init__(theta)
        self._simulate_native_quant = True
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        self.fake_quant = fake_quant
        self.experimental_mm_cache_size = experimental_mm_cache_size
        self.experimental_mm_cache_sets = experimental_mm_cache_sets
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)

        # Input premultiplier.
        self.premul_input = theta.optional_tensor("premul_input")
        self.q_input: Optional[QuantizerTensor] = theta.optional_tensor("q_input")
        self.qdq_input: Optional[QuantizedTensor] = theta.optional_tensor("qdq_input")
        if self.q_input is not None and self.qdq_input is not None:
            raise AssertionError(f"LinearLayer cannot have both q_input and qdq_input")
        self.qdq_output: Optional[QuantizedTensor] = theta.optional_tensor("qdq_output")

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        q_input = self.q_input
        qdq_input = self.qdq_input
        qdq_output = self.qdq_output
        if self.premul_input is not None:
            x = ops.elementwise(torch.mul, x, self.premul_input)

        if q_input is not None:
            x = q_input.quantize(x)
            if self.fake_quant:
                x = x.unpack().dequant()

        elif qdq_input is not None:
            x = qdq_input.quantize(x).unpack().dequant()

        # TODO - This should be removed once the compiler supports it:
        if (
            self.experimental_mm_cache_sets is not None
            and self.experimental_mm_cache_size is not None
        ):
            contract = x.shape[-1]
            if isinstance(x, SplitPrimitiveTensor):
                contract = contract // x.shard_count
            element_size = torch.finfo(x.dtype).bits
            cache_line_size = self.experimental_mm_cache_size * 8 // element_size
            cache_size = self.experimental_mm_cache_sets * cache_line_size
            if contract % cache_size == 0:
                x = ops.pad(x, [0, cache_line_size], constant=0, per_shard=True)
                weight = ops.pad(
                    weight, [0, cache_line_size], constant=0, per_shard=True
                )

        y = ops.linear(x, weight, bias)

        # Unconditionally dequantize.
        if isinstance(y, QuantizedTensor):
            y = y.unpack().dequant()
        # Note that f8_e4m3fnuz types on AMD GPUs accumulate to fp32.
        # We can truncate to fp16 in iree, so we do a cast here
        # to account for this in the IR. This is may not be the right
        # level to do this, but for now its here.
        if not isinstance(y, QuantizedTensor):
            if y.dtype == torch.float8_e4m3fnuz:
                y = ops.to(y, torch.float16)
                return y
        if qdq_output is not None:
            y = qdq_output.quantize(y).unpack().dequant()
        return y
