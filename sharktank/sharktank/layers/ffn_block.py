# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from sharktank import ops
from sharktank.types import AnyTensor

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
        rms_epsilon: float | None = None,
        is_gated: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        activation_dtype: Optional[torch.dtype] = torch.float16,
        fake_quant: bool = False,
        add_residual: bool = True,
    ):
        """
        add_residual:
            At the end of the block add to the input.
        """
        super().__init__(theta)

        self.is_gated = is_gated
        self.activation_fn = activation_fn
        self.ffn_norm = torch.nn.Identity()
        self.add_residual = add_residual

        if self.is_gated:
            self.add_module(
                "ffn_gate", LinearLayer(theta("ffn_gate"), fake_quant=fake_quant)
            )
        if "ffn_norm" in theta:
            # Llama & MoE models
            self.ffn_norm = RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)
        elif "layer_norm" in theta:
            # T5 model
            self.ffn_norm = RMSNormLayer(
                theta("layer_norm"), epsilon=rms_epsilon, dtype=activation_dtype
            )

        self.add_module("ffn_up", LinearLayer(theta("ffn_up"), fake_quant=fake_quant))
        self.add_module(
            "ffn_down", LinearLayer(theta("ffn_down"), fake_quant=fake_quant)
        )

    def forward(
        self,
        h: AnyTensor,
    ) -> AnyTensor:

        h_norm = self.ffn_norm(h)
        if self.is_gated:
            ffn_gate = ops.elementwise(self.activation_fn, self.ffn_gate(h_norm))
            ffn_up = self.ffn_up(h_norm)
            ffn_down = self.ffn_down(ffn_gate * ffn_up)
        else:
            ffn_up = self.ffn_up(h_norm)
            ffn_act = ops.elementwise(self.activation_fn, ffn_up)
            ffn_down = self.ffn_down(ffn_act)

        if self.add_residual:
            return h + ffn_down
        return ffn_down
