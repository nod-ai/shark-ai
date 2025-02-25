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
    ):
        super().__init__(theta)
        self._simulate_native_quant = True
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        self.fake_quant = fake_quant
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

        y = ops.linear(x, weight, bias)
        # Unconditionally dequantize.
        if isinstance(y, QuantizedTensor):
            y = y.unpack().dequant()
        if qdq_output is not None:
            y = qdq_output.quantize(y).unpack().dequant()
        return y
