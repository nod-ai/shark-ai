# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from sharktank.utils._helpers import run_iree_vs_torch_fx


class Linear(torch.nn.Module):
    def __init__(self, in_f, out_f, bias=False, dtype=torch.float32):
        super().__init__()
        self.lin = torch.nn.Linear(in_f, out_f, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.lin(x)

@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.float16, 1e-4)])
def test_linear_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)
    m = Linear(64, 64, bias=False, dtype=dtype)
    x = torch.randn(2, 8, 64, dtype=dtype)
    run_iree_vs_torch_fx(m, args=(x,), atol=atol, rtol=0)
