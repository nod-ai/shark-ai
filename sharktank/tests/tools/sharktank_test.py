# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from subprocess import check_call
from pathlib import Path
import pytest

from sharktank.layers import model_config_presets
from sharktank.utils import chdir


@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dummy_model")


def test_list():
    check_call(["sharktank", "list"])


def test_show():
    check_call(["sharktank", "show", "dummy-model-local-llvm-cpu"])


def test_export_compile(dummy_model_path: Path):
    with chdir(dummy_model_path):
        check_call(["sharktank", "export", "dummy-model-local-llvm-cpu"])
        check_call(["sharktank", "compile", "dummy-model-local-llvm-cpu"])
        from .. import models

        assert model_config_presets[
            "dummy-model-local-llvm-cpu"
        ].export_parameters_path.exists()
