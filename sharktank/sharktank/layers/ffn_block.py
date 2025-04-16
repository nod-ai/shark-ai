# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import torch.nn.functional as F
from .. import ops
from ..types import AnyTensor

from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer


__all__ = [
    "FFN",
]


class FFN(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        rms_epsilon: float,
        is_gated: bool = True,
        activation_fn: Callable[[AnyTensor], AnyTensor] = F.silu,
        fake_quant: bool = False,
    ):
        super().__init__(theta)

        self.is_gated = is_gated
        self.activation_fn = activation_fn

        ffn_suffix = ""
        if theta.optional_tensor("ffn_gate_shexp") is not None:
            ffn_suffix = "_shexp"

        ffn_gate = "ffn_gate" + ffn_suffix
        ffn_up = "ffn_up" + ffn_suffix
        ffn_down = "ffn_down" + ffn_suffix

        if self.is_gated:
            self.add_module(
                "ffn_gate", LinearLayer(theta(ffn_gate), fake_quant=fake_quant)
            )
        if theta.optional_tensor("ffn_norm") is not None:
            self.ffn_norm = RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)

        self.add_module("ffn_up", LinearLayer(theta(ffn_up), fake_quant=fake_quant))
        self.add_module("ffn_down", LinearLayer(theta(ffn_down), fake_quant=fake_quant))

    def forward(
        self,
        h: AnyTensor,
    ) -> AnyTensor:
        h_norm = h
        if self.ffn_norm:
            h_norm = self.ffn_norm(h)
        if self.is_gated:
            ffn_gate = ops.elementwise(self.activation_fn, self.ffn_gate(h_norm))
            ffn_up = self.ffn_up(h_norm)
            ffn_down = self.ffn_down(ffn_gate * ffn_up)
        else:
            ffn_up = self.ffn_up(h_norm)
            ffn_activation = ops.elementwise(self.activation_fn, ffn_up)
            ffn_down = self.ffn_down(ffn_activation)

        return h + ffn_down
