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
from sharktank.utils._helpers import run_iree_vs_torch_fx


class TokenEmbeddingSmall(torch.nn.Module):
    def __init__(self, vocab_size=128, hidden=64, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden, dtype=dtype))
        self.dtype = dtype
    def forward(self, ids: torch.Tensor):
        return self.weight[ids]


@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-4)])
def test_token_embedding_iree_vs_eager(request, dtype, atol):
    torch.manual_seed(42)
    
    # Get IRPA path from command line argument
    irpa_path = request.config.getoption("--parameters")
    
    # Skip test if no IRPA path provided
    if irpa_path is None:
        pytest.skip("No IRPA path provided. Use --parameters to specify the IRPA file.")
    
    # Skip test if IRPA file doesn't exist
    if not Path(irpa_path).exists():
        pytest.skip(f"IRPA file not found: {irpa_path}")
    
    dataset = Dataset.load(irpa_path)
    m = TokenEmbeddingLayer(dataset.root_theta("token_embd"), dtype=dtype)
    inp_tensors = torch.randint(0, 128, (2, 8), dtype=torch.long)
    run_iree_vs_torch_fx(m, args=(inp_tensors,), atol=atol, rtol=0.0, parameters_path=irpa_path)


@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-4)])
def test_token_embedding_mock_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)

    # Each test assumes all inputs are in the correct dtype 
    # as that information is required to export the model
    m = TokenEmbeddingSmall(vocab_size=128, hidden=64, dtype=dtype)
    inp_tensors = torch.randint(0, 128, (2, 8), dtype=torch.long)
    run_iree_vs_torch_fx(m, args=(inp_tensors,), atol=atol, rtol=0.0)


if __name__ == "__main__":
    test_token_embedding_mock_iree_vs_eager()
    print("TokenEmbeddingLayer mock test complete!")