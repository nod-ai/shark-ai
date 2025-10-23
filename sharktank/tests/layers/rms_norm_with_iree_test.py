# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
import gc
from sharktank.utils.iree import run_iree_vs_torch_eager
from sharktank.utils._iree_compile_flags_config import LLM_HIP_COMPILE_FLAGS
from sharktank.utils.testing import is_hip_condition


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden=64, eps=1e-5, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden, dtype=dtype))
        self.eps = eps
        self.dtype = dtype

    def forward(self, x):
        # y = x * weight / sqrt(mean(x^2) + eps)
        var = (x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps)
        y = x * inv
        return y * self.weight  # broadcast over last dim


@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.bfloat16, 1e-2)])
def test_rms_norm_mock_iree_vs_eager(dtype, atol):
    gc.collect()
    torch.manual_seed(42)
    m = RMSNorm(hidden=64, dtype=dtype)
    x = torch.randn(2, 8, 64, dtype=dtype)
    run_iree_vs_torch_eager(
        m, input_args=(x,), atol=atol, rtol=0, compile_flags=LLM_HIP_COMPILE_FLAGS
    )
