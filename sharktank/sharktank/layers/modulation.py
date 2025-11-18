# Copyright 2024 Black Forest Labs. Inc. and Flux Authors
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Modulation Layer adapted from black-forest-labs' flux implementation
https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
"""

import torch
import torch.nn.functional as F

from amdsharktank import ops
from amdsharktank.types import Theta
from .base import ThetaLayer
from .linear import LinearLayer


class ModulationOut:
    def __init__(self, shift, scale, gate):
        self.shift = shift
        self.scale = scale
        self.gate = gate


class ModulationLayer(ThetaLayer):
    def __init__(self, theta: Theta, double: bool):
        super().__init__(theta)

        self.is_double = double
        self.multiplier = 6 if double else 3
        self.add_module("lin", LinearLayer(theta("lin")))

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        silu_result = ops.elementwise(F.silu, vec)
        out = self.lin(silu_result)[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
