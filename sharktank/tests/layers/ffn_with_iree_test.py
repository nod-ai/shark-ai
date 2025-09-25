# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch
import pytest
from sharktank.utils._helpers import run_iree_vs_torch_fx

class FFN(torch.nn.Module):
    def __init__(self, hidden=64, inter=128, dtype=torch.float32, activation="silu"):
        super().__init__()
        self.w_up = torch.nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.w_gate = torch.nn.Linear(hidden, inter, bias=False, dtype=dtype)
        self.w_down = torch.nn.Linear(inter, hidden, bias=False, dtype=dtype)
        self.activation = activation

    def forward(self, x):
        if self.activation == "silu":
            return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))
        else:
            return self.w_down(torch.nn.functional.gelu(self.w_up(x)))

@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.float16, 1e-4)])
def test_ffn_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)
    m = FFN(hidden=64, inter=128, dtype=dtype, activation="silu")
    x = torch.randn(2, 8, 64, dtype=dtype)
    run_iree_vs_torch_fx(m, args=(x,), atol=atol, rtol=0)
