# Table of Contents
1. [CK Flash Attention](#ck-flash-attention)
    - [Setup](#setup)
    - [Sample Benchmark](#testing-ck-performance)

# CK Flash Attention

## Setup

### Clone Repository
```shell
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout ck_tile
git submodule update --init --recursive --progress
```

### Create Python Environment
Create a Python virtual environment and install the required torch dependencies:

```shell
python3 -m venv .venv
source .venv/bin/activate

# Install torch compiled with ROCm v6.1 
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1
pip install pytest wheel packaging
```

### Verify ROCm HIP Toolkit
Ensure that ROCm HIP toolkit version 6.1 or above is installed on your system and set as the default. Verify this by running:

```shell
hipcc --version
```

You should see output similar to:
```
HIP version: 6.2.41133-ba05059c0
...
```

If multiple versions are installed, set the appropriate environment variables:

```shell
# Replace 6.1.x with the appropriate version
export ROCM_PATH="/opt/rocm-6.1.x"
export PATH="/opt/rocm-6.1.x/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm-6.1.x/lib:$LD_LIBRARY_PATH"
```

### Run Setup Script
For MI300 GPUs, run the setup script with:

```shell
GPU_ARCHS=gfx942 python setup.py install
```

## Testing CK Performance

Use the following Python script to test the performance of CK Flash Attention.

```python
import torch
import time
from flash_attn import flash_attn_func

device = "cuda:0"
dtype = torch.bfloat16

batch_size = 1
seqlen_q = 376
seqlen_kv = 64296
nheads = 42
headdim = 64

q = torch.rand([batch_size, seqlen_q, nheads, headdim], dtype=dtype, device=device)
k = torch.rand([batch_size, seqlen_kv, nheads, headdim], dtype=dtype, device=device)
v = torch.rand([batch_size, seqlen_kv, nheads, headdim], dtype=dtype, device=device)

class TestModule(torch.nn.Module):
    def forward(self, q, k, v):
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

torch._dynamo.config.suppress_errors = True
test_module = torch.compile(TestModule(), dynamic=True)
test_module.to(device=device, dtype=dtype)

warmup_steps = 10
for _ in range(warmup_steps):
    test_module(q, k, v)
torch.cuda.synchronize()

eval_steps = 1000
start_time = time.time()
for _ in range(eval_steps):
    test_module(q, k, v)
torch.cuda.synchronize()

avg_time = (time.time() - start_time) / eval_steps
print(f"Average time per iteration: {avg_time:.6f} seconds")
```
