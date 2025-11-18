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
from amdsharktank.layers import BaseLayer
from amdsharktank.utils import debugging
from amdsharktank.utils.patching import (
    FilterKind,
    PatchFilterElement,
    TraceTensorModulePatch,
)
from amdsharktank.utils.testing import TempDirTestBase


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
            self.some_other_method(arg0)
            return arg0, arg1

        def some_other_method(self, arg0: torch.Tensor):
            """Just some other that `forward` method to test if it gets traced or not
            correctly."""
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
    return outer


def test_trace_tensor_module_patch(config_tracing, module_for_patching: BaseLayer):
    tensor0 = torch.arange(1, 3, dtype=int)
    tensor1 = torch.arange(3, 6, dtype=int)

    patcher = TraceTensorModulePatch(with_before_call=True)
    patcher.patch_child_modules(module_for_patching)

    # Args passed like that to test positional and kwargs get correct trace names.
    module_for_patching(
        tensor0,
        arg1=tensor1,
    )

    path_expected_value_map = {
        debugging.flags.trace_path / f".forward.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f".forward.%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.%1.safetensors": tensor1,
        debugging.flags.trace_path / f"inner.forward.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"inner.forward.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f"inner.forward.%0.safetensors": tensor0,
        debugging.flags.trace_path / f"inner.forward.%1.safetensors": tensor1,
    }
    for path, expected_value in path_expected_value_map.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
            torch.testing.assert_close(recorded_tensor, expected_value, rtol=0, atol=0)


def test_trace_tensor_module_with_regex_filter(
    config_tracing, module_for_patching: BaseLayer
):
    tensor0 = torch.arange(1, 3, dtype=int)
    tensor1 = torch.arange(3, 6, dtype=int)

    patcher = TraceTensorModulePatch(with_before_call=True)

    filter = [
        PatchFilterElement(regex=re.escape("inner.forward"), kind=FilterKind.EXCLUDE),
        PatchFilterElement(regex=".*\\.forward"),
        PatchFilterElement(regex=".+\\.some_other_method"),
    ]
    patcher.patch_child_modules(module_for_patching, filter=filter)

    # Args passed like that to test positional and kwargs get correct trace names.
    module_for_patching(
        tensor0,
        arg1=tensor1,
    )

    path_expected_value_map = {
        debugging.flags.trace_path / f".forward.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f".forward.%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.%1.safetensors": tensor1,
        debugging.flags.trace_path
        / f"inner.some_other_method.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"inner.some_other_method.%0.safetensors": tensor0,
    }
    for path, expected_value in path_expected_value_map.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
            torch.testing.assert_close(recorded_tensor, expected_value, rtol=0, atol=0)

    expected_excluded_paths = [
        debugging.flags.trace_path / "inner.forward.%0.safetensors",
        debugging.flags.trace_path / "inner.forward.%1.safetensors",
        debugging.flags.trace_path / "inner.forward.arg%0.safetensors",
        debugging.flags.trace_path / "inner.forward.arg%arg1.safetensors",
    ]
    for p in expected_excluded_paths:
        assert not p.exists()


def test_trace_tensor_module_with_fnmatch_filter(
    config_tracing, module_for_patching: BaseLayer
):
    tensor0 = torch.arange(1, 3, dtype=int)
    tensor1 = torch.arange(3, 6, dtype=int)

    patcher = TraceTensorModulePatch(with_before_call=True)

    filter = [
        PatchFilterElement(fnmatch="inner.forward", kind=FilterKind.EXCLUDE),
        PatchFilterElement(fnmatch=".forward"),
        PatchFilterElement(fnmatch="*.some_other_method"),
    ]
    patcher.patch_child_modules(module_for_patching, filter=filter)

    # Args passed like that to test positional and kwargs get correct trace names.
    module_for_patching(
        tensor0,
        arg1=tensor1,
    )

    path_expected_value_map = {
        debugging.flags.trace_path / f".forward.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.arg%arg1.safetensors": tensor1,
        debugging.flags.trace_path / f".forward.%0.safetensors": tensor0,
        debugging.flags.trace_path / f".forward.%1.safetensors": tensor1,
        debugging.flags.trace_path
        / f"inner.some_other_method.arg%0.safetensors": tensor0,
        debugging.flags.trace_path / f"inner.some_other_method.%0.safetensors": tensor0,
    }
    for path, expected_value in path_expected_value_map.items():
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
            torch.testing.assert_close(recorded_tensor, expected_value, rtol=0, atol=0)

    expected_excluded_paths = [
        debugging.flags.trace_path / "inner.forward.%0.safetensors",
        debugging.flags.trace_path / "inner.forward.%1.safetensors",
        debugging.flags.trace_path / "inner.forward.arg%0.safetensors",
        debugging.flags.trace_path / "inner.forward.arg%arg1.safetensors",
    ]
    for p in expected_excluded_paths:
        assert not p.exists()
