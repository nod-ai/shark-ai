# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import iree.runtime
import pytest
import platform
import torch

from parameterized import parameterized
from pathlib import Path
from amdsharktank.layers import create_model, model_config_presets
from amdsharktank.types import DefaultPrimitiveTensor
from amdsharktank.utils import chdir
from amdsharktank.utils.iree import (
    device_array_to_host,
    get_iree_devices,
    run_model_with_iree_run_module,
    tensor_to_device_array,
    trace_model_with_tracy,
    with_iree_device_context,
)
from amdsharktank.utils.testing import skip
from amdsharktank.models.dummy import DummyModel
from amdsharktank import ops
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
