# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import iree.runtime
import pytest
import platform
import torch
import torch.nn as nn

from parameterized import parameterized
from pathlib import Path
from sharktank.layers import create_model, model_config_presets
from sharktank.types import DefaultPrimitiveTensor
from sharktank.utils import chdir
from sharktank.utils.iree import (
    adapt_torch_module_to_iree,
    compile_torch_module_to_iree,
    device_array_to_host,
    get_iree_devices,
    oneshot_iree_run,
    run_model_with_iree_run_module,
    tensor_to_device_array,
    trace_model_with_tracy,
    with_iree_device_context,
)
from sharktank.utils.testing import skip
from sharktank.models.dummy import DummyModel
from sharktank import ops
from unittest import TestCase


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dummy_model")


@pytest.fixture(scope="session")
def dummy_model(dummy_model_path: Path) -> DummyModel:
    with chdir(dummy_model_path):
        model = create_model(model_config_presets["dummy-model-local-llvm-cpu"])
        model.export()
        model.compile()
        return model


def test_run_model_with_iree_run_module(
    dummy_model: DummyModel, dummy_model_path: Path
):
    with chdir(dummy_model_path):
        run_model_with_iree_run_module(dummy_model.config, function="forward_bs1")


@skip(
    reason=(
        "The test hangs. Probably during compilation or IREE module "
        "execution. We can't determine easily what is going on as running "
        "tests in parallel with pyest-xdist is incompatible with capture "
        "disabling with --capture=no. No live logs are available from the CI."
        " TODO: investigate"
    )
)
@pytest.mark.xfail(
    platform.system() == "Windows",
    raises=FileNotFoundError,
    reason="The Python package for Windows does not include iree-tracy-capture.",
)
def test_trace_model_with_tracy(dummy_model: DummyModel, dummy_model_path: Path):
    with chdir(dummy_model_path):
        trace_path = Path(f"{dummy_model.config.iree_module_path}.tracy")
        assert not trace_path.exists()
        trace_model_with_tracy(dummy_model.config, function="forward_bs1")
        assert trace_path.exists()


@pytest.mark.usefixtures("iree_flags")
class TestTensorConversion(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    @parameterized.expand(
        [
            (torch.float32, torch.float32),
            (torch.float64, torch.float64),
            (torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fnuz, torch.float32),
        ]
    )
    def testRoundtrip(self, dtype: torch.dtype, dtype_for_equality_check: torch.dtype):
        if dtype.is_floating_point:
            tensor = torch.rand([3, 4], dtype=torch.float32).to(dtype=dtype)
        else:
            tensor = torch.randint(low=0, high=100, size=[3, 4], dtype=dtype)

        iree_devices = get_iree_devices(device=self.iree_device, device_count=1)

        def roundtrip(iree_devices: list[iree.runtime.HalDevice]):
            tensor_roundtrip = device_array_to_host(
                tensor_to_device_array(tensor, iree_devices[0])
            )
            assert tensor.to(dtype=dtype_for_equality_check).equal(
                tensor_roundtrip.to(dtype=dtype_for_equality_check)
            )

        with_iree_device_context(roundtrip, iree_devices)

    def testTensorToDeviceArraySupportsDefaultPrimitiveTensor(self):
        tensor = DefaultPrimitiveTensor(data=torch.arange(1, 4, dtype=int))

        iree_devices = get_iree_devices(device=self.iree_device, device_count=1)

        def roundtrip(iree_devices: list[iree.runtime.HalDevice]):
            tensor_roundtrip = device_array_to_host(
                tensor_to_device_array(tensor, iree_devices[0])
            )
            assert ops.equal(tensor, tensor_roundtrip)

        with_iree_device_context(roundtrip, iree_devices)


COMPILE_FLAGS = [
    "--iree-hal-target-device=local",
    "--iree-hal-local-target-device-backends=llvm-cpu",
]


class SimpleModel(nn.Module):
    def forward(self, x):
        return x + 1


class MultiOutputModel(nn.Module):
    def forward(self, x):
        return x + 1, x * 2


class TestCompileTorchModule:
    """Tests for compile_torch_module_to_iree."""

    def test_compilation(self, tmp_path):
        """Test compilation with optional artifact saving."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)
        mlir_path = tmp_path / "model.mlir"
        vmfb_path = tmp_path / "model.vmfb"

        vmfb_bytes = compile_torch_module_to_iree(
            model,
            example_args=(example_input,),
            compile_args=COMPILE_FLAGS,
            save_mlir_to=mlir_path,
            save_vmfb_to=vmfb_path,
        )

        assert len(vmfb_bytes) > 100

        assert mlir_path.exists()
        assert mlir_path.stat().st_size > 0

        assert vmfb_path.exists()
        assert vmfb_path.stat().st_size > 0


class TestAdaptTorchModuleToIree:
    """Tests for adapt_torch_module_to_iree."""

    def test_basic_loading_and_execution(self, deterministic_random_seed):
        """Test that loaded module executes and produces correct output shape."""
        model = SimpleModel()
        example_input = torch.randint(0, 100, (2, 32), dtype=torch.int64)

        iree_module = adapt_torch_module_to_iree(
            model,
            example_args=(example_input,),
            device="local-sync",
            compile_args=COMPILE_FLAGS,
        )

        result = iree_module.forward(example_input)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 32)

    def test_output_matches_torch(self, deterministic_random_seed):
        """Test that IREE output matches torch output"""
        model = SimpleModel()
        model.eval()
        example_input = torch.randint(0, 100, (2, 32), dtype=torch.int64)

        torch_output = model(example_input)
        iree_module = adapt_torch_module_to_iree(
            model,
            example_args=(example_input,),
            device="local-sync",
            compile_args=COMPILE_FLAGS,
        )
        iree_output = iree_module.forward(example_input)

        assert torch.equal(iree_output, torch_output)

    def test_multi_output_model(self):
        """Test model with multiple outputs."""
        model = MultiOutputModel()
        example_input = torch.randn(2, 32)

        iree_module = adapt_torch_module_to_iree(
            model,
            example_args=(example_input,),
            device="local-sync",
            compile_args=COMPILE_FLAGS,
        )

        result = iree_module.forward(example_input)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 32)
        assert result[1].shape == (2, 32)


class TestOneshotCompileAndRun:
    """Tests for oneshot_iree_run."""

    def test_basic_oneshot(self, deterministic_random_seed):
        """Test basic one-shot execution."""
        model = SimpleModel()
        example_input = torch.randint(0, 100, (2, 32), dtype=torch.int64)

        result = oneshot_iree_run(
            model,
            args=(example_input,),
            device="local-sync",
            compile_args=COMPILE_FLAGS,
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 32)

    def test_oneshot_matches_torch(self, deterministic_random_seed):
        """Test that one-shot execution matches torch."""
        model = SimpleModel()
        model.eval()
        example_input = torch.randint(0, 100, (2, 32), dtype=torch.int64)

        torch_output = model(example_input)
        iree_output = oneshot_iree_run(
            model,
            args=(example_input,),
            device="local-sync",
            compile_args=COMPILE_FLAGS,
        )

        # Use exact comparison for integer arithmetic
        assert torch.equal(iree_output, torch_output)
