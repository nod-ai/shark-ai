# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from subprocess import check_call
from pathlib import Path
import pytest

from amdsharktank.layers import model_config_presets, register_all_models, ModelConfig
from amdsharktank.utils import chdir


@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dummy_model")


def test_list():
    check_call(["amdshark", "model", "list"])


def test_show():
    check_call(["amdshark", "model", "show", "dummy-model-local-llvm-cpu"])


def test_export_compile(dummy_model_path: Path):
    with chdir(dummy_model_path):
        check_call(["amdshark", "model", "export", "dummy-model-local-llvm-cpu"])
        check_call(["amdshark", "model", "compile", "dummy-model-local-llvm-cpu"])

        register_all_models()
        config = ModelConfig.create(
            **model_config_presets["dummy-model-local-llvm-cpu"]
        )
        assert config.export_parameters_path.exists()
