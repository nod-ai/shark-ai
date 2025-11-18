# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pytest

from typing import Generator
from logging import Logger
from unittest.mock import MagicMock

from iree.compiler import ir  # type: ignore

from amdsharktuner import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    mock_logger = MagicMock(spec=Logger)
    with common.TunerContext(logger=mock_logger) as ctx:
        yield ctx


@pytest.fixture
def mlir_ctx() -> Generator[ir.Context, None, None]:
    with ir.Context() as ctx:
        yield ctx
