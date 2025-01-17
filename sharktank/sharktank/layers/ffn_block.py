# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Callable

import torch
import torch.nn.functional as F
from .. import ops
from ..types import AnyTensor

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
        activation_fn: Callable[[AnyTensor], AnyTensor] = F.silu,
        experimental_mm_cache_size: Optional[int] = None,
        experimental_mm_cache_sets: Optional[int] = None,
    ):
        super().__init__(theta)

        self.is_gated = is_gated
        self.activation_fn = activation_fn
        if self.is_gated:
            self.add_module(
                "ffn_gate",
                LinearLayer(
                    theta("ffn_gate"),
                    experimental_mm_cache_size=experimental_mm_cache_size,
                    experimental_mm_cache_sets=experimental_mm_cache_sets,
                ),
            )
        self.add_module(
            "ffn_up",
            LinearLayer(
                theta("ffn_up"),
                experimental_mm_cache_size=experimental_mm_cache_size,
                experimental_mm_cache_sets=experimental_mm_cache_sets,
            ),
        )
        self.add_module(
            "ffn_down",
            LinearLayer(
                theta("ffn_down"),
                experimental_mm_cache_size=experimental_mm_cache_size,
                experimental_mm_cache_sets=experimental_mm_cache_sets,
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
            return ffn_down
        else:
            h = self.ffn_up(h)
            h = ops.elementwise(self.activation_fn, h)
            h = self.ffn_down(h)
            return h
