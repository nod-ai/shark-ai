# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from unittest import TestCase

pytest_plugins = "pytester"


def test_function_scope_iree_flags_fixture_smoke(iree_flags):
    """Make sure fixtures work on function scope, not just class scope."""
    assert hasattr(iree_flags, "iree_device")
    assert hasattr(iree_flags, "iree_hip_target")
    assert hasattr(iree_flags, "iree_hal_target_device")
    assert hasattr(iree_flags, "iree_hal_local_target_device_backends")


def test_function_scope_model_artifacts_fixture_smoke(model_artifacts):
    """Make sure fixtures work on function scope, not just class scope."""
    assert "llama3_8b_tokenizer_path" in model_artifacts


@pytest.mark.usefixtures("iree_flags", "model_artifacts")
class TestFixtureOnTestCase(TestCase):
    """Make sure fixtures work on TestCase classes."""

    def test_iree_flags_smoke(self):
        assert hasattr(self, "iree_device")
        assert hasattr(self, "iree_hip_target")
        assert hasattr(self, "iree_hal_target_device")
        assert hasattr(self, "iree_hal_local_target_device_backends")

    def test_model_artifacts_smoke(self):
        # Check one sample value.
        assert hasattr(self, "llama3_8b_tokenizer")
