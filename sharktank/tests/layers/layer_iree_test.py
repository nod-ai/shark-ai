# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
import torch
import torch.nn as nn


class TinyLinear(nn.Module):
    def __init__(self, in_features=16, out_features=8, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

# --- Utilities: compile with iree-turbine and run with iree.runtime ---
def iree_compile_and_load(module: nn.Module, example_input: torch.Tensor, driver: str = "local-task"):
    """
    Exports `module` with iree-turbine AOT API, compiles to a binary (vmfb in-memory),
    and returns a callable for the entry function.

    CPU driver: driver='local-task' (default)
    For GPU later, you can use: 'hip://0', 'vulkan', etc., and pass extra compile args.
    """
    import iree.runtime as ireert
    import iree.turbine.aot as aot

    # 1) Export (simple API). Under the covers uses torch-mlir via turbine.
    export_output = aot.export(module, example_input)  # Simple “one-shot” export

    # 2) Compile to a deployable artifact (vmfb in-memory)
    #    You can pass extra_args if needed; start with defaults for CPU.
    binary = export_output.compile(save_to=None)

    # 3) Load with IREE runtime and return the entrypoint (Py func-like)
    config = ireert.Config(driver)  # 'local-task' → CPU
    vm_module = ireert.load_vm_module(
        ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
        config,
    )

    # The default entry is usually "forward" for AOT exports.
    # On turbine's "simple" export, it is often materialized as `main` or `forward`.
    # We try both for convenience.
    entry = getattr(vm_module, "forward", None) or getattr(vm_module, "main")
    return entry

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

@pytest.mark.parametrize("in_features,out_features,bias", [(16, 8, True), (32, 32, False)])
@pytest.mark.parametrize("batch", [1, 4])
def test_tinylinear_torch_vs_iree(in_features, out_features, bias, batch):
    torch.manual_seed(0)

    # Arrange
    layer = TinyLinear(in_features, out_features, bias=bias).eval()
    x = torch.randn(batch, in_features)

    # Baseline in PyTorch
    with torch.no_grad():
        y_ref = layer(x)

    # IREE AOT compile + run on CPU (local-task)
    entry = iree_compile_and_load(layer, x, driver="local-task")  # CPU path
    # IREE runtime expects numpy inputs
    y_iree = entry(to_numpy(x)).to_host()  # returns iree runtime NdArray-like → to_host() → numpy

    # Assert shape/dtype equivalence
    assert list(y_ref.shape) == list(y_iree.shape)
    assert str(y_ref.dtype) in ("torch.float32",)  # default path uses f32
    assert y_iree.dtype == np.float32

    # Numeric tolerance
    np.testing.assert_allclose(to_numpy(y_ref), y_iree, rtol=1e-5, atol=2e-5)
