# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from subprocess import check_call
from pathlib import Path
import pytest

from sharktank.layers import model_config_presets, register_all_models, ModelConfig
from sharktank.utils import chdir


@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dummy_model")


def test_list():
    check_call(["shark", "model", "list"])


def test_show():
    check_call(["shark", "model", "show", "dummy-model-local-llvm-cpu"])


def test_export_compile(dummy_model_path: Path):
    with chdir(dummy_model_path):
        check_call(["shark", "model", "export", "dummy-model-local-llvm-cpu"])
        check_call(["shark", "model", "compile", "dummy-model-local-llvm-cpu"])
        from .. import models

        register_all_models()
        config = ModelConfig.create(
            **model_config_presets["dummy-model-local-llvm-cpu"]
        )
        assert config.export_parameters_path.exists()
