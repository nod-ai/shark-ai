# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch
import pytest
from sharktank.utils.iree import run_iree_vs_torch_eager
from sharktank.utils._iree_compile_flags_config import LLM_HIP_COMPILE_FLAGS
from sharktank.utils.testing import is_hip_condition


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


@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize(
    "dtype,atol",
    [
        (torch.float32, 1e-4),
        pytest.param(
            torch.float16,
            1e-4,
            marks=pytest.mark.xfail(
                reason="Numerical mismatch error - https://github.com/nod-ai/shark-ai/issues/2415"
            ),
        ),
    ],
)
def test_ffn_mock_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)
    m = FFN(hidden=64, inter=128, dtype=dtype, activation="silu")
    x = torch.randn(2, 8, 64, dtype=dtype)
    run_iree_vs_torch_eager(
        m, input_args=(x,), atol=atol, rtol=0, compile_flags=LLM_HIP_COMPILE_FLAGS
    )
