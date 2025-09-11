# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import re
import safetensors
import torch

from pathlib import Path
from sharktank.layers import BaseLayer
from sharktank.utils import debugging
from sharktank.utils.patching import (
    FilterKind,
    PatchFilterElement,
    TraceTensorModulePatch,
)
from sharktank.utils.testing import TempDirTestBase


@pytest.fixture
def config_tracing(tmp_path: Path):
    # setup
    callback_stash = debugging.get_trace_tensor_callback()
    debugging.set_trace_tensor_callback(debugging.trace_tensor_to_safetensors_callback)

    enable_tensor_trace_stash = debugging.flags.enable_tensor_trace
    debugging.flags.enable_tensor_trace = True

    trace_path_stash = debugging.flags.trace_path
    debugging.flags.trace_path = tmp_path

    yield

    # teardown
    debugging.set_trace_tensor_callback(callback_stash)
    debugging.flags.enable_tensor_trace = enable_tensor_trace_stash
    debugging.flags.trace_path = trace_path_stash


@pytest.fixture
def module_for_patching() -> BaseLayer:
    class Inner(BaseLayer):
        def forward(
            self, arg0: torch.Tensor, arg1: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.other_method(arg0)
            return arg0, arg1

        def other_method(self, arg0: torch.Tensor):
            return arg0

    class Outer(BaseLayer):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(
            self, arg0: torch.Tensor, arg1: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.inner(arg0, arg1=arg1)

    outer = Outer()
    outer.trace_tensor_key_prefix = "outer."
    outer.set_recursively_submodules_default_trace_tensor_key_prefix()
    return outer


def test_trace_tensor_module_patch(config_tracing, module_for_patching: BaseLayer):
    tensor0 = torch.arange(1, 3, dtype=int)
    tensor1 = torch.arange(3, 6, dtype=int)

    patcher = TraceTensorModulePatch(with_before_call=True)
    patcher.patch_child_modules(module_for_patching)

    module_for_patching(tensor0, arg1=tensor1)

    path_expected_value_map = {
        debugging.flags.trace_path / f"outer.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f"outer.%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.%1.safetensors": tensor1,
        debugging.flags.trace_path / f"outer.inner.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.inner.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f"outer.inner.%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.inner.%1.safetensors": tensor1,
    }
    for path, expected_value in path_expected_value_map.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
            torch.testing.assert_close(recorded_tensor, expected_value, rtol=0, atol=0)


def test_trace_tensor_module_patch_more_complex_fully_qualified_name_filter(
    config_tracing, module_for_patching: BaseLayer
):
    tensor0 = torch.arange(1, 3, dtype=int)
    tensor1 = torch.arange(3, 6, dtype=int)

    patcher = TraceTensorModulePatch(with_before_call=True)

    filter = [
        PatchFilterElement(
            regex=re.escape("outer.inner.forward"), kind=FilterKind.EXCLUDE
        ),
        PatchFilterElement(regex=re.escape(".+\.forward")),
        PatchFilterElement(regex=re.escape("outer\..+\.other_method")),
    ]
    patcher.patch_child_modules(module_for_patching, filter=filter)
    module_for_patching(tensor0, arg1=tensor1)

    path_expected_value_map = {
        debugging.flags.trace_path / f"outer.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f"outer.%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.%1.safetensors": tensor1,
        debugging.flags.trace_path / f"outer.inner.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"outer.inner.%0.safetensors": tensor0,
    }
    for path, expected_value in path_expected_value_map.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
            torch.testing.assert_close(recorded_tensor, expected_value, rtol=0, atol=0)
