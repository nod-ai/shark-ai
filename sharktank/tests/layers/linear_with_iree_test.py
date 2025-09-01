# sharktank/tests/layers/linear_with_iree_test.py
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
    torch.manual_seed(0)
    m = Linear(64, 64, bias=False, dtype=dtype)
    x = torch.randn(2, 8, 64, dtype=dtype)
    run_iree_vs_torch_fx(m, args=(x,), atol=atol, rtol=0)
