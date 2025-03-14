# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from shortfin.support.deps import ShortfinDepNotFoundError


@pytest.fixture(autouse=True)
def require_deps():
    try:
        import shortfin_apps.llm
    except ShortfinDepNotFoundError as e:
        pytest.skip(f"Dep not available: {e}")


import shortfin as sf


@pytest.fixture(scope="module")
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture(scope="module")
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture(scope="module")
def device(fiber):
    return fiber.device(0)
