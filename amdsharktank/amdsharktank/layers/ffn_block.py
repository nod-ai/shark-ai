# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import torch
import torch.nn.functional as F
from amdsharktank import ops
from amdsharktank.types import AnyTensor

from .base import Theta, ThetaLayer
from .linear import LinearLayer


__all__ = [
    "FFN",
]


class FFN(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        is_gated: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        fake_quant: bool = False,
        matmul_kernel: str | None = None,
    ):
        """
        add_residual:
            At the end of the block add to the input.
        """
        super().__init__(theta)

        self.is_gated = is_gated
        self.activation_fn = activation_fn

        if self.is_gated:
            self.add_module(
                "ffn_gate",
                LinearLayer(
                    theta("ffn_gate"),
                    fake_quant=fake_quant,
                    matmul_kernel=matmul_kernel,
                ),
            )

        self.add_module(
            "ffn_up",
            LinearLayer(
                theta("ffn_up"),
                fake_quant=fake_quant,
                matmul_kernel=matmul_kernel,
            ),
        )
        self.add_module(
            "ffn_down",
            LinearLayer(
                theta("ffn_down"),
                fake_quant=fake_quant,
                matmul_kernel=matmul_kernel,
            ),
        )

    def forward(
        self,
        h: AnyTensor,
    ) -> AnyTensor:
        if self.is_gated:
            ffn_gate = ops.elementwise(self.activation_fn, self.ffn_gate(h))
            ffn_up = self.ffn_up(h)
            ffn_down = self.ffn_down(ffn_gate * ffn_up)
        else:
            ffn_up = self.ffn_up(h)
            ffn_activation = ops.elementwise(self.activation_fn, ffn_up)
            ffn_down = self.ffn_down(ffn_activation)

        return ffn_down
