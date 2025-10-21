# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from pathlib import Path
from sharktank.layers.token_embedding import TokenEmbeddingLayer
from sharktank.types.theta import Dataset
from sharktank.utils.iree import (
    run_iree_vs_torch_eager,
)
from sharktank.utils._iree_compile_flags_config import LLM_HIP_COMPILE_FLAGS
from sharktank.utils.testing import is_hip_condition, validate_and_get_irpa_path


class TokenEmbeddingSmall(torch.nn.Module):
    def __init__(self, vocab_size=128, hidden=64, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden, dtype=dtype))
        self.dtype = dtype

    def forward(self, ids: torch.Tensor):
        return self.weight[ids]


@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-4)])
def test_token_embedding_iree_vs_eager(request, dtype, atol):
    torch.manual_seed(42)

    # Validate and get IRPA path
    irpa_path = validate_and_get_irpa_path(request)

    dataset = Dataset.load(irpa_path)
    m = TokenEmbeddingLayer(dataset.root_theta("token_embd"), dtype=dtype)
    inp_tensors = torch.randint(0, 128, (2, 8), dtype=torch.long)
    run_iree_vs_torch_eager(
        m,
        input_args=(inp_tensors,),
        atol=atol,
        rtol=0.0,
        compile_flags=LLM_HIP_COMPILE_FLAGS,
        parameters_path=irpa_path,
    )


@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-4)])
def test_token_embedding_mock_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)

    # Each test assumes all inputs are in the correct dtype
    # as that information is required to export the model
    m = TokenEmbeddingSmall(vocab_size=128, hidden=64, dtype=dtype)
    inp_tensors = torch.randint(0, 128, (2, 8), dtype=torch.long)
    run_iree_vs_torch_eager(
        m,
        input_args=(inp_tensors,),
        atol=atol,
        rtol=0.0,
        compile_flags=LLM_HIP_COMPILE_FLAGS,
    )


if __name__ == "__main__":
    test_token_embedding_mock_iree_vs_eager()
    print("TokenEmbeddingLayer mock test complete!")
