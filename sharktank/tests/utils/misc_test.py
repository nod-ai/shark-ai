# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from amdsharktank.utils import iterables_equal
from collections.abc import Iterable


@pytest.mark.parametrize(
    "iterable1, iterable2, expected",
    [
        ([1, 2], [1, 2], True),
        ([1, 3], [1, 2], False),
        ([1, 2, 3], [1, 2], False),
    ],
)
def test_iterables_equal(iterable1: Iterable, iterable2: Iterable, expected: bool):
    result = iterables_equal(iterable1, iterable2)
    assert expected == result

    # Mirror
    result = iterables_equal(iterable2, iterable1)
    assert expected == result
