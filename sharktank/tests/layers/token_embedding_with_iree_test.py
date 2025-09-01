# sharktank/tests/layers/token_embedding_with_iree_test.py
import torch
import pytest
from sharktank.utils._helpers import run_iree_vs_torch_fx
from sharktank.layers.token_embedding import TokenEmbeddingLayer

class TokenEmbeddingSmall(torch.nn.Module):
    def __init__(self, vocab_size=128, hidden=64, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden, dtype=dtype))
        self.dtype = dtype
    def forward(self, ids: torch.Tensor):
        return self.weight[ids]

@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.float16, 1e-4)])
def test_token_embedding_iree_vs_eager(dtype, atol):
    torch.manual_seed(0)

    # Each test assumes all inputs are in the correct dtype 
    # as that information is required to export the model
    m = TokenEmbeddingSmall(vocab_size=128, hidden=64, dtype=dtype)

    from pathlib import Path
    irpa_path = Path('/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa')
    from sharktank.types.theta import Dataset
    dataset = Dataset.load(irpa_path)

    m = TokenEmbeddingLayer(dataset.root_theta("token_embd"), dtype=dtype)
    # ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    inp_tensors = torch.randint(0, 128, (2, 8), dtype=torch.long)
    run_iree_vs_torch_fx(m, args=(inp_tensors,), atol=atol, rtol=0.0)
