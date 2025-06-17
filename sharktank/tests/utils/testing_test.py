# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from sharktank.utils.testing import xfail

pytest_plugins = "pytester"


def test_strict_xfail_with_successful_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=True, match="test_xfail_with_successful_match")
        def test_f():
            raise RuntimeError("test_xfail_with_successful_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_failed_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=True, match="string_that_can_not_be_found")
        def test_f():
            raise RuntimeError("test_xfail_with_failed_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_non_strict_xfail_with_failed_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=False, match="string_that_can_not_be_found")
        def test_f():
            raise RuntimeError("test_xfail_with_failed_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_without_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=True)
        def test_f():
            raise RuntimeError("test_xfail_without_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_wrong_exception(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=True)
        def test_f():
            raise ValueError("")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_match_with_multiple_lines_in_exception_string(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=RuntimeError, strict=True, match="line2")
        def test_f():
            raise RuntimeError("line1\\nline2\\nline3")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_xfail_xpass_with_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(match="match")
        def test_f():
            pass
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xpassed=1)


def test_multiple_strict_xfails_with_successful_match(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(raises=ValueError, strict=True, match="match")
        @xfail(raises=ValueError, strict=True, match="match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_successful_match_and_false_condition(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(condition=False, raises=ValueError, reason="", strict=True, match="match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_with_failed_match_and_true_condition(pytester: pytest.Pytester):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(condition=True, raises=ValueError, reason="", strict=True, match="not a match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_multiple_strict_xfails_with_failed_match_and_false_condition_in_first_xfail(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(condition=True, raises=ValueError, reason="", strict=True, match="match")
        @xfail(condition=False, raises=ValueError, reason="", strict=True, match="not a match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_multiple_strict_xfails_with_failed_match_and_true_condition_in_first_xfail(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(condition=True, raises=ValueError, reason="", strict=True, match="match")
        @xfail(condition=True, raises=ValueError, reason="", strict=True, match="not a match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_multiple_non_strict_xfails_with_false_condition_in_first_xfail(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(strict=False)
        @xfail(condition=False, strict=False, raises=ValueError, reason="")
        def test_f():
            raise RuntimeError("")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_xfail_failed_match_does_not_interfere_with_exception_message_of_failed_previous_match(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(strict=True, match="This is the exception")
        @xfail(raises=ValueError, match="not a match")
        def test_f():
            raise ValueError("This is the exception")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_xfail_failed_match_does_not_interfere_with_exception_message_of_failed_previous_match(
    pytester: pytest.Pytester,
):
    pytester.makepyfile(
        """
        from sharktank.utils.testing import xfail

        @xfail(strict=True, match="This is the exception")
        @xfail(raises=ValueError, match="not a match")
        def test_f():
            raise ValueError("This is the exception")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
