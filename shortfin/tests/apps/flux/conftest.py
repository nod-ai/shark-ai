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
        import shortfin_apps.sd
    except ShortfinDepNotFoundError as e:
        pytest.skip(f"Dep not available: {e}")
