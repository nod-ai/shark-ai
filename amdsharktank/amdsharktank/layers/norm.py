# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from amdsharktank import ops
from amdsharktank.types import Theta
from typing import Optional
from .base import BaseLayer, ThetaLayer


class RMSNormLayer(ThetaLayer):
    """Computes the unbiased full RMS layer normalization.

    Because normalization is sensitive to floating point error, we support
    an explicit dtype that the input will be casted to prior to performing
    the compute. The result will be cast back to the input dtype.
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        epsilon: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.epsilon = epsilon
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x = ops.to(x, self.dtype)
        norm = ops.rms_norm(x, self.weight, epsilon=self.epsilon, orig_dtype=orig_dtype)
        # Will automatically upcast to the dtype of the weight, which is
        # often in higher precision. Downcast back to expected.
        norm = ops.to(norm, orig_dtype)
        return norm


class LayerNorm(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        bias_name: str = "bias",
        eps: float = 1e-05,
        normalized_shape: Optional[tuple[int]] = None,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor):
        return ops.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            normalized_shape=self.normalized_shape,
        )
        return ops.layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class L2Norm(BaseLayer):
    def __init__(self, dim: int | tuple[int, ...], epsilon: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(self.dim, keepdim=True) + self.epsilon)

    def forward(self, x):
        return self._norm(x.to(dtype=torch.float32)).type_as(x)
