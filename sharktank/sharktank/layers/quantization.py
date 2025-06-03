# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import torch
from .base import Theta, ThetaLayer
from sharktank.types import AnyTensor, QuantizerTensor

__all__ = [
    "QuantizationLayer",
]


class QuantizationLayer(ThetaLayer):
    """A layer that performs quantization on its input tensor.
    
    This layer applies quantization using a Quantizer from the theta,
    which can be useful for adding explicit quantization points in models
    or for integration with quantization frameworks like Brevitas.
    """

    def __init__(
        self,
        theta: Theta,
        *,
        quantizer_name: str = "quantizer",
        enabled = True,
    ):
        super().__init__(theta)
        self.quantizer: Optional[QuantizerTensor] = theta.optional_tensor(quantizer_name)
        self.enabled = enabled
        
        if enabled and self.quantizer is None:
            raise ValueError(f"QuantizationLayer requires a quantizer named '{quantizer_name}' in theta")

    def forward(self, x: AnyTensor) -> AnyTensor:
        """Apply quantization to the input tensor."""
        if not self.enabled or self.quantizer is None:
            # Pass through if no quantizer
            return x
        
        # Quantize the input
        return self.quantizer.quantize(x)