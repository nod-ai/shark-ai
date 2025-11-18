# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
from amdsharktank import ops
from .base import Theta, ThetaLayer
from amdsharktank.types import QuantizedTensor, QuantizerTensor, ShardedTensor

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
        matmul_kernel: Optional[str] = None,
    ):
        super().__init__(theta)
        self._simulate_native_quant = True
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        self.fake_quant = fake_quant
        self.matmul_kernel = matmul_kernel
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)

        # Input premultiplier.
        self.premul_input = theta.optional_tensor("premul_input")
        self.q_input: Optional[QuantizerTensor] = theta.optional_tensor("q_input")
        self.qdq_input: Optional[QuantizedTensor] = theta.optional_tensor("qdq_input")
        if self.q_input is not None and self.qdq_input is not None:
            raise AssertionError(f"LinearLayer cannot have both q_input and qdq_input")
        self.qdq_output: Optional[QuantizedTensor] = theta.optional_tensor("qdq_output")
        self.q_output: Optional[QuantizerTensor] = theta.optional_tensor("q_output")

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        q_input = self.q_input
        qdq_input = self.qdq_input
        qdq_output = self.qdq_output
        if self.premul_input is not None:
            x = ops.elementwise(torch.mul, x, self.premul_input)

        if q_input is not None:
            x = ops.quantize(x, q_input)
            if self.fake_quant:
                x = ops.dequantize(x)

        elif qdq_input is not None:
            x = ops.dequantize(ops.quantize(x, qdq_input))

        y = ops.linear(x, weight, bias, matmul_impl=self.matmul_kernel)

        if isinstance(y, QuantizedTensor) or (
            isinstance(y, ShardedTensor) and isinstance(y.shards[0], QuantizedTensor)
        ):
            y = ops.dequantize(y)

        if qdq_output is not None:
            y = ops.dequantize(ops.quantize(y, qdq_output))
        return y
