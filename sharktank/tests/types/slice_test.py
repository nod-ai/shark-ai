# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from amdsharktank.types import canonicalize_slice_descriptor, CanonicalSlice, Slice
from amdsharktank.utils.misc import iterables_equal


class TestSlice:
    @pytest.mark.parametrize(
        "slice_, expected_slice",
        [
            (None, (None, slice(0, 2, 1), slice(0, 3, 1))),
            (slice(None), (slice(0, 2, 1), slice(0, 3, 1))),
            (slice(-2, -1, 2), (slice(0, 1, 2), slice(0, 3, 1))),
            ([1, 2], ([1, 2], slice(0, 3, 1))),
            (1, (1, slice(0, 3, 1))),
            (-1, (1, slice(0, 3, 1))),
            ([-2, 1], ([0, 1], slice(0, 3, 1))),
            ((None, slice(None), None, [1, 2]), (None, slice(0, 2, 1), None, [1, 2])),
        ],
    )
    def test_canonicalize_slice_descriptor(
        self, slice_: Slice, expected_slice: CanonicalSlice
    ):
        shape = [2, 3]
        actual = canonicalize_slice_descriptor(slice_, shape)
        assert iterables_equal(actual, expected_slice)

    @pytest.mark.parametrize(
        "slice_, expected_exception",
        [
            (-5, IndexError),
            ([0, -5], IndexError),
            (slice(-5), IndexError),
            (slice(1, -5), IndexError),
        ],
    )
    def test_canonicalize_slice_descriptor_exception(
        self, slice_: Slice, expected_exception: Exception
    ):
        shape = [2, 3]
        with pytest.raises(expected_exception):
            canonicalize_slice_descriptor(slice_, shape)
