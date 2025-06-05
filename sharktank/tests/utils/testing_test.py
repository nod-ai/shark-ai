# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from sharktank.utils.testing import xfail, XfailMatchError


def test_xfail_with_successful_match():
    @xfail(raises=RuntimeError, strict=True, match="test_xfail_with_successful_match")
    def f():
        raise RuntimeError("test_xfail_with_successful_match")

    with pytest.raises(RuntimeError, match="test_xfail_with_successful_match"):
        f()


def test_xfail_with_failed_match():
    @xfail(raises=RuntimeError, strict=True, match="string_that_can_not_be_found")
    def f():
        raise RuntimeError("test_xfail_with_failed_match")

    with pytest.raises(
        XfailMatchError,
        match='Failed to match error "test_xfail_with_failed_match" '
        'against expected match "string_that_can_not_be_found"',
    ):
        f()


def test_xfail_without_match():
    @xfail(raises=RuntimeError, strict=True)
    def f():
        raise RuntimeError("test_xfail_without_match")

    with pytest.raises(RuntimeError, match="test_xfail_without_match"):
        f()


def test_xfail_match_with_multiple_lines_in_exception_string():
    @xfail(raises=RuntimeError, strict=True, match="line2")
    def f():
        raise RuntimeError("line1\nline2\nline3")

    with pytest.raises(RuntimeError, match="line1\nline2\nline3"):
        f()
