# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import pytest
import torch
import re

from pathlib import Path
from amdsharktank.utils.testing import assert_tensor_close

pytest_plugins = "pytester"


@pytest.fixture
def pytester_with_conftest(pytester: pytest.Pytester) -> pytest.Pytester:
    """Copy our conftest.py into the test dir so that Pytester tests can pick it up."""
    with open(f"{Path(__file__).parent.parent.parent / 'conftest.py'}", "r") as f:
        pytester.makeconftest(f.read())
    return pytester


class TestAssertTensorClose:
    def test_equal_tree(self):
        dtype = torch.int32
        expected = [
            {
                "a": [
                    torch.tensor([1, 2], dtype=dtype),
                    torch.tensor([3, 4], dtype=dtype),
                ],
                "b": torch.tensor([5, 6], dtype=dtype),
            }
        ]
        actual = copy.deepcopy(expected)
        assert_tensor_close(actual, expected, rtol=0, atol=0)

    def test_tree_structure_not_equal(self):
        dtype = torch.int32
        expected = [
            {
                "a": [
                    torch.tensor([1, 2], dtype=dtype),
                    torch.tensor([3, 4], dtype=dtype),
                ],
                "b": torch.tensor([5, 6], dtype=dtype),
            }
        ]
        actual = copy.deepcopy(expected)
        del actual[0]["b"]
        with pytest.raises(AssertionError, match="Tree structure not equal"):
            assert_tensor_close(actual, expected)

    def test_tensor_in_tree_not_equal(self):
        dtype = torch.int32
        expected = [
            {
                "a": [
                    torch.tensor([1, 2], dtype=dtype),
                    torch.tensor([3, 4], dtype=dtype),
                ],
                "b": torch.tensor([5, 6], dtype=dtype),
            }
        ]
        actual = copy.deepcopy(expected)
        actual[0]["a"][0][0] = 10
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Trees not equal: elements in trees with path (0, 'a', 0) not equal"
            ),
        ):
            assert_tensor_close(actual, expected)

    def test_tensor_equal_with_non_tensor(self):
        dtype = torch.int32
        expected = [
            {
                "a": [
                    torch.tensor([1, 2], dtype=dtype),
                    None,
                ],
                "b": torch.tensor([5, 6], dtype=dtype),
            }
        ]
        actual = copy.deepcopy(expected)
        assert_tensor_close(actual, expected)

    def test_tensor_not_equal_with_non_tensor(self):
        dtype = torch.int32
        expected = [
            {
                "a": [
                    torch.tensor([1, 2], dtype=dtype),
                    torch.tensor(3, dtype=dtype),
                ],
                "b": torch.tensor([5, 6], dtype=dtype),
            }
        ]
        actual = copy.deepcopy(expected)
        actual[0]["a"][1] = object()
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Trees not equal: elements in trees with path (0, 'a', 1) not equal"
            ),
        ):
            assert_tensor_close(actual, expected)


def test_deterministic_random_seed(deterministic_random_seed):
    """Make the deterministic_random_seed fixture never changes RNG generation compared
    to the expected.
    This should not change across different minor versions. It likely would not change
    across major version bumps."""
    import torch
    import numpy
    import random

    torch_actual = torch.randint(high=99999999, size=[1], dtype=torch.int32)
    torch_expected = torch.tensor([57136067], dtype=torch.int32)
    torch.testing.assert_close(torch_actual, torch_expected, atol=0, rtol=0)

    numpy_actual = numpy.random.randint(
        low=0, high=99999999, size=[1], dtype=numpy.int32
    )
    numpy_expected = numpy.array([75434668], dtype=numpy.int32)
    numpy.testing.assert_equal(numpy_actual, numpy_expected)

    builtin_actual = random.randint(0, 99999999)
    builtin_expected = 51706749
    assert builtin_actual == builtin_expected


def test_strict_xfail_with_successful_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            raises=RuntimeError, strict=True, match="test_xfail_with_successful_match"
        )
        def test_f():
            raise RuntimeError("test_xfail_with_successful_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_failed_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            raises=RuntimeError, strict=True, match="string_that_can_not_be_found"
        )
        def test_f():
            raise RuntimeError("test_xfail_with_failed_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_non_strict_xfail_with_failed_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(raises=RuntimeError, strict=False, match="string_that_can_not_be_found")
        def test_f():
            raise RuntimeError("test_xfail_with_failed_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_without_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(raises=RuntimeError, strict=True)
        def test_f():
            raise RuntimeError("test_xfail_without_match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_wrong_exception(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(raises=RuntimeError, strict=True)
        def test_f():
            raise ValueError("")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_match_with_multiple_lines_in_exception_string(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(raises=RuntimeError, strict=True, match="line2")
        def test_f():
            raise RuntimeError("line1\\nline2\\nline3")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_xfail_xpass_with_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(match="match")
        def test_f():
            pass
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xpassed=1)


def test_multiple_strict_xfails_with_successful_match(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(raises=ValueError, strict=True, match="match")
        @pytest.mark.xfail(raises=ValueError, strict=True, match="match")
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_strict_xfail_with_successful_match_and_false_condition(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            condition=False, raises=ValueError, reason="", strict=True, match="match"
        )
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_strict_xfail_with_failed_match_and_true_condition(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            condition=True,
            raises=ValueError,
            reason="",
            strict=True,
            match="not a match"
        )
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_multiple_strict_xfails_with_failed_match_and_false_condition_in_first_xfail(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            condition=True, raises=ValueError, reason="", strict=True, match="match"
        )
        @pytest.mark.xfail(
            condition=False,
            raises=ValueError,
            reason="",
            strict=True,
            match="not a match"
        )
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_multiple_strict_xfails_with_failed_match_and_true_condition_in_first_xfail(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(
            condition=True, raises=ValueError, reason="", strict=True, match="match"
        )
        @pytest.mark.xfail(
            condition=True,
            raises=ValueError,
            reason="",
            strict=True,
            match="not a match"
        )
        def test_f():
            raise ValueError("match")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_multiple_non_strict_xfails_with_false_condition_in_first_xfail(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(strict=False)
        @pytest.mark.xfail(condition=False, strict=False, raises=ValueError, reason="")
        def test_f():
            raise RuntimeError("")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(xfailed=1)


def test_xfail_failed_match_does_not_interfere_with_exception_message_of_failed_previous_match(
    pytester_with_conftest: pytest.Pytester,
):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(strict=True, match="This is the exception")
        @pytest.mark.xfail(raises=ValueError, match="not a match")
        def test_f():
            raise ValueError("This is the exception")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)


def test_parametrized_xfail_with_failed_match(pytester_with_conftest: pytest.Pytester):
    pytester = pytester_with_conftest
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize(
            "a", [pytest.param(1, marks=pytest.mark.xfail(match="not a match"))]
        )
        def test_f(a: int):
            raise ValueError("This is the exception")
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
