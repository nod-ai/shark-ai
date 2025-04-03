# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
import random

from typing import Any, List

import shortfin as sf
import shortfin.array as sfnp


@pytest.fixture
def lsys():
    # TODO: Port this test to use memory type independent access. It currently
    # presumes unified memory.
    # sc = sf.SystemBuilder()
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber):
    return fiber.device(0)


def test_argmax(device):
    src = sfnp.device_array(device, [4, 16, 128], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod([1, 16, 128]))]
    for i in range(4):
        src.view(i).items = data
        data.reverse()

    # default variant
    result = sfnp.argmax(src)
    assert result.shape == [4, 16]
    assert result.view(0).items.tolist() == [127] * 16
    assert result.view(1).items.tolist() == [0] * 16
    assert result.view(2).items.tolist() == [127] * 16
    assert result.view(3).items.tolist() == [0] * 16

    # keepdims variant
    result = sfnp.argmax(src, keepdims=True)
    assert result.shape == [4, 16, 1]

    # out= variant
    out = sfnp.device_array(device, [4, 16], dtype=sfnp.int64)
    sfnp.argmax(src, out=out)
    assert out.shape == [4, 16]
    assert out.view(0).items.tolist() == [127] * 16
    assert out.view(1).items.tolist() == [0] * 16
    assert out.view(2).items.tolist() == [127] * 16
    assert out.view(3).items.tolist() == [0] * 16

    # out= keepdims variant (left aligned rank broadcast is allowed)
    out = sfnp.device_array(device, [4, 16, 1], dtype=sfnp.int64)
    sfnp.argmax(src, keepdims=True, out=out)
    assert out.shape == [4, 16, 1]
    assert out.view(0).items.tolist() == [127] * 16
    assert out.view(1).items.tolist() == [0] * 16
    assert out.view(2).items.tolist() == [127] * 16
    assert out.view(3).items.tolist() == [0] * 16


def test_argmax_axis0(device):
    src = sfnp.device_array(device, [4, 16], dtype=sfnp.float32)
    for j in range(4):
        src.view(j).items = [
            float((j + 1) * (i + 1) - j * 4) for i in range(math.prod([1, 16]))
        ]
    print(repr(src))

    # default variant
    result = sfnp.argmax(src, axis=0)
    assert result.shape == [16]
    assert result.items.tolist() == [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    # keepdims variant
    result = sfnp.argmax(src, axis=0, keepdims=True)
    assert result.shape == [1, 16]

    # out= variant
    out = sfnp.device_array(device, [16], dtype=sfnp.int64)
    sfnp.argmax(src, axis=0, out=out)

    # out= keepdims variant
    out = sfnp.device_array(device, [1, 16], dtype=sfnp.int64)
    sfnp.argmax(src, axis=0, keepdims=True, out=out)


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_argmax_dtypes(device, dtype):
    # Just verifies that the dtype functions. We don't have IO support for
    # some of these.
    src = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    sfnp.argmax(src)


@pytest.mark.parametrize(
    "k,axis",
    [
        # Min sort, default axis
        [3, None],
        # Min sort, axis=-1
        [20, -1],
        # Max sort, default axis
        [-3, None],
        # Max sort, axis=-1
        [-20, -1],
    ],
)
def test_argpartition(device, k, axis):
    src = sfnp.device_array(device, [1, 1, 128], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod([1, 1, 128]))]
    randomized_data = data[:]
    random.shuffle(randomized_data)
    src.items = randomized_data

    result = (
        sfnp.argpartition(src, k) if axis is None else sfnp.argpartition(src, k, axis)
    )

    assert result.shape == src.shape

    expected_values = data[:k] if k >= 0 else data[k:]

    k_slice = slice(0, k) if k >= 0 else slice(k, None)

    indices = result.view(0, 0, k_slice).items.tolist()
    values = [randomized_data[index] for index in indices]
    assert sorted(values) == sorted(expected_values)


def test_argpartition_large_array(device):
    """Test to ensure that `argpartition` doesn't hang on large `device_arrays`.

    Accuracy validation is handled by `test_log_softmax` above.
    """
    src = sfnp.device_array(device, [1, 1, 128256], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    k = -1000
    sfnp.argpartition(src, k)


def test_argpartition_out_variant(device):
    k, axis = -3, -1
    src = sfnp.device_array(device, [1, 1, 128], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]

    randomized_data = data[:]
    random.shuffle(randomized_data)
    src.items = randomized_data

    output_array = sfnp.device_array(device, src.shape, dtype=sfnp.int64)
    result_out = sfnp.argpartition(src, k, axis, out=output_array)
    result_no_out = sfnp.argpartition(src, k, axis)

    assert result_out.shape == src.shape
    out_items = result_out.items.tolist()
    no_out_items = result_no_out.items.tolist()
    assert out_items == no_out_items


def test_argpartition_axis0(device):
    def _get_top_values_by_col_indices(
        indices: List[int], data: List[List[int]], k: int
    ) -> List[List[int]]:
        """Obtain the top-k values from out matrix, using column indices.

        For this test, we partition by column (axis == 0). This is just some
        helper logic to obtain the values from the original matrix, given
        then column indices.

        Args:
            indices (List[int]): Flattened indices from `sfnp.argpartition`
            data (List[List[int]]): Matrix containing original values.
            k (int): Specify top-k values to select.

        Returns:
            List[List[int]]: Top-k values for each column.
        """
        num_cols = len(data[0])

        top_values_by_col = []

        for c in range(num_cols):
            # Collect the row indices for the first k entries in column c.
            col_row_idxs = [indices[r * num_cols + c] for r in range(k)]

            # Map those row indices into actual values in `data`.
            col_values = [data[row_idx][c] for row_idx in col_row_idxs]

            top_values_by_col.append(col_values)

        return top_values_by_col

    def _get_top_values_by_sorting(
        data: List[List[float]], k: int
    ) -> List[List[float]]:
        """Get the top-k value for each col in the matrix, using sorting.

        This is just to obtain a comparison for our `argpartition` testing.

        Args:
            data (List[List[int]]): Matrix of data.
            k (int): Specify top-k values to select.

        Returns:
            List[List[float]]: Top-k values for each column.
        """
        num_rows = len(data)
        num_cols = len(data[0])

        top_values_by_col = []

        for c in range(num_cols):
            # Extract the entire column 'c' into a list
            col = [data[r][c] for r in range(num_rows)]
            # Sort the column in ascending order
            col_sorted = sorted(col)
            # The first k elements are the k smallest
            col_k_smallest = col_sorted[:k]
            top_values_by_col.append(col_k_smallest)

        return top_values_by_col

    k, axis = 2, 0
    src = sfnp.device_array(device, [3, 4], dtype=sfnp.float32)
    # data = [[float(i) for i in range(math.prod(src.shape))]]
    data = [[i for i in range(src.shape[-1])] for _ in range(src.shape[0])]
    for i in range(len(data)):
        random.shuffle(data[i])

    for i in range(src.shape[0]):
        src.view(i).items = data[i]

    result = sfnp.argpartition(src, k, axis)
    assert result.shape == src.shape

    expected_values = _get_top_values_by_sorting(data, k)
    top_values = _get_top_values_by_col_indices(result.items.tolist(), data, k)
    for result, expected in zip(top_values, expected_values):
        assert sorted(result) == sorted(expected)


def test_argpartition_error_cases(device):
    # Invalid `input` dtype
    with pytest.raises(
        ValueError,
    ):
        src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.int64)
        sfnp.argpartition(src, 0)

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    # Invalid `axis`
    with pytest.raises(
        ValueError,
    ):
        sfnp.argpartition(src, 1, 3)
        sfnp.argpartition(src, 1, -4)

    # Invalid `k`
    with pytest.raises(
        ValueError,
    ):
        sfnp.argpartition(src, 17)
        sfnp.argpartition(src, -17)

    # Invalid `out` dtype
    with pytest.raises(
        ValueError,
    ):
        out = sfnp.device_array(device, src.shape, dtype=sfnp.float32)
        sfnp.argpartition(src, 2, -1, out)


def approximately_equal(a: Any, b: Any, rel_tol=1e-2, abs_tol=0.0) -> bool:
    """
    Recursively checks if two nested lists (or scalar values) are approximately equal.

    Args:
        a: First list or scalar.
        b: Second list or scalar.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.

    Returns:
        True if all corresponding elements are approximately equal.
    """
    # If both are lists, iterate element-wise
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(
            approximately_equal(sub_a, sub_b, rel_tol, abs_tol)
            for sub_a, sub_b in zip(a, b)
        )

    # Otherwise, assume they are scalars and compare
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


test_cases = [
    # Single values should always return `0.0`
    {"shape": [1, 1], "data": [[420.0]], "axis": -1, "expected": [[0.0]]},
    {"shape": [1, 1], "data": [[420.0]], "axis": None, "expected": [[0.0]]},
    # Two values with constant offset of `1` should always return
    # 0th: -log(1 + e)
    # 1st: 1 - log(1 + e)
    {
        "shape": [1, 2],
        "data": [[float(42), float(43)]],
        "axis": -1,
        "expected": [
            [
                -math.log(1 + math.e),
                1 - math.log(1 + math.e),
            ]
        ],
    },
    {
        "shape": [1, 2],
        "data": [[float(42), float(43)]],
        "axis": None,
        "expected": [
            [
                -math.log(1 + math.e),
                1 - math.log(1 + math.e),
            ]
        ],
    },
    # When given uniform values, each item should be equal to -log(n), where
    # n is the size of the targeted axis.
    {
        "shape": [5, 10],
        "data": [[float(42) for _ in range(10)] for _ in range(5)],
        "axis": -1,
        "expected": [[-math.log(10) for _ in range(10)] for _ in range(5)],
    },
    {
        "shape": [5, 10],
        "data": [[float(42) for _ in range(10)] for _ in range(5)],
        "axis": None,
        "expected": [[-math.log(10) for _ in range(10)] for _ in range(5)],
    },
    # Axis 0 test. If given all uniform values, and taking column-wise
    # log_softmax, then each item should be equal to -log(2).
    {
        "shape": [2, 3],
        "data": [[float(42) for _ in range(3)] for _ in range(2)],
        "axis": 0,
        "expected": [[-math.log(2) for _ in range(3)] for _ in range(2)],
    },
]


@pytest.mark.parametrize("params", test_cases)
def test_log_softmax(device, params):
    shape = params["shape"]
    data = params["data"]
    axis = params["axis"]
    expected = params["expected"]
    src = sfnp.device_array(device, shape, dtype=sfnp.float32)
    for i in range(len(data)):
        src.view(i).items = data[i]
    if axis is not None:
        result = sfnp.log_softmax(src, axis)
    else:
        result = sfnp.log_softmax(src)
    results = []
    for i in range(len(data)):
        vals = result.view(i).items.tolist()
        results.append(vals)
    assert approximately_equal(results, expected)


def test_log_softmax_large_array(device):
    """Test to ensure that `log_softmax` doesn't hang on large `device_arrays`.

    Accuracy validation is handled by `test_log_softmax` above.
    """
    src = sfnp.device_array(device, [1, 1, 128256], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    sfnp.log_softmax(src)


def test_log_softmax_out_variant(device):
    axis = -1
    src = sfnp.device_array(device, [1, 1, 128], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    output_array = sfnp.device_array(device, src.shape, dtype=sfnp.float32)
    result_out = sfnp.log_softmax(src, axis, out=output_array)
    result_no_out = sfnp.log_softmax(src, axis)

    assert result_out.shape == src.shape
    assert result_out.dtype.name == src.dtype.name
    out_items = result_out.items.tolist()
    no_out_items = result_no_out.items.tolist()
    assert approximately_equal(out_items, no_out_items)


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_log_softmax_dtype(device, dtype):
    src = sfnp.device_array(device, [1, 16, 128], dtype=dtype)
    sfnp.log_softmax(src)


def test_log_softmax_error_cases(device):
    # Invalid `input` dtype
    with pytest.raises(
        ValueError,
    ):
        src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.int64)
        sfnp.log_softmax(src)

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    # Invalid `axis`
    with pytest.raises(
        ValueError,
    ):
        sfnp.log_softmax(src, 3)
        sfnp.log_softmax(src, -4)

    # Invalid `out` dtype
    with pytest.raises(
        ValueError,
    ):
        out = sfnp.device_array(device, src.shape, dtype=sfnp.float16)
        sfnp.log_softmax(src, -1, out)


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_argpartition_dtypes(device, dtype):
    src = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    sfnp.argpartition(src, 0)


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_fill_randn_default_generator(device, dtype):
    out1 = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    with out1.map(write=True) as m:
        m.fill(bytes(1))
    sfnp.fill_randn(out1)
    out2 = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    with out2.map(write=True) as m:
        m.fill(bytes(1))
    sfnp.fill_randn(out2)

    with out1.map(read=True) as m1, out2.map(read=True) as m2:
        # The default generator should populate two different arrays.
        contents1 = bytes(m1)
        contents2 = bytes(m2)
        assert contents1 != contents2


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
    ],
)
def test_fill_randn_explicit_generator(device, dtype):
    gen1 = sfnp.RandomGenerator(42)
    gen2 = sfnp.RandomGenerator(42)
    out1 = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    with out1.map(write=True) as m:
        m.fill(bytes(1))
    sfnp.fill_randn(out1, generator=gen1)
    out2 = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    with out2.map(write=True) as m:
        m.fill(bytes(1))
    sfnp.fill_randn(out2, generator=gen2)
    zero = sfnp.device_array(device, [4, 16, 128], dtype=dtype)
    with zero.map(write=True) as m:
        m.fill(bytes(1))

    with out1.map(read=True) as m1, out2.map(read=True) as m2, zero.map(
        read=True
    ) as mz:
        # Using explicit generators with the same seed should produce the
        # same distributions.
        contents1 = bytes(m1)
        contents2 = bytes(m2)
        assert contents1 == contents2
        # And not be zero.
        assert contents1 != bytes(mz)


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.uint8,
        sfnp.uint16,
        sfnp.uint32,
        sfnp.uint64,
        sfnp.int8,
        sfnp.int16,
        sfnp.int32,
        sfnp.int64,
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
        sfnp.float64,
    ],
)
def test_convert(device, dtype):
    input_array = sfnp.device_array(device, [2, 3], dtype=sfnp.int32)
    with input_array.map(write=True) as m:
        m.fill(16)
    intermediate = sfnp.convert(input_array, dtype=dtype)
    with input_array.map(write=True) as m:
        m.fill(0)
    sfnp.convert(intermediate, out=input_array)
    assert list(input_array.items) == 6 * [16]


def round_half_up(n):
    return math.floor(n + 0.5)


def round_half_away_from_zero(n):
    rounded_abs = round_half_up(abs(n))
    return math.copysign(rounded_abs, n)


@pytest.mark.parametrize(
    "dtype,sfnp_func,ref_round_func",
    [
        (sfnp.bfloat16, sfnp.round, round_half_away_from_zero),
        (sfnp.float16, sfnp.round, round_half_away_from_zero),
        (sfnp.float32, sfnp.round, round_half_away_from_zero),
        (sfnp.bfloat16, sfnp.ceil, math.ceil),
        (sfnp.float16, sfnp.ceil, math.ceil),
        (sfnp.float32, sfnp.ceil, math.ceil),
        (sfnp.bfloat16, sfnp.floor, math.floor),
        (sfnp.float16, sfnp.floor, math.floor),
        (sfnp.float32, sfnp.floor, math.floor),
        (sfnp.bfloat16, sfnp.trunc, math.trunc),
        (sfnp.float16, sfnp.trunc, math.trunc),
        (sfnp.float32, sfnp.trunc, math.trunc),
    ],
)
def test_nearest_int_no_conversion(device, dtype, sfnp_func, ref_round_func):
    input = sfnp.device_array(device, [2, 3], dtype=dtype)
    sfnp.fill_randn(input)
    ref_rounded = [
        ref_round_func(n) for n in sfnp.convert(input, dtype=sfnp.float32).items
    ]
    output = sfnp_func(input)
    assert output.dtype == dtype
    output_items = sfnp.convert(output, dtype=sfnp.float32).items
    print(output_items)
    for ref, actual in zip(ref_rounded, output_items):
        assert ref == pytest.approx(actual)


@pytest.mark.parametrize(
    "dtype,out_dtype,sfnp_func,ref_round_func",
    [
        # Round
        (sfnp.float16, sfnp.int8, sfnp.round, round_half_away_from_zero),
        (sfnp.float32, sfnp.int8, sfnp.round, round_half_away_from_zero),
        (sfnp.float32, sfnp.int16, sfnp.round, round_half_away_from_zero),
        (sfnp.float32, sfnp.int32, sfnp.round, round_half_away_from_zero),
        # Note that we do not test unsigned conversion with random data.
        # Ceil
        (sfnp.float16, sfnp.int8, sfnp.ceil, math.ceil),
        (sfnp.float32, sfnp.int8, sfnp.ceil, math.ceil),
        (sfnp.float32, sfnp.int16, sfnp.ceil, math.ceil),
        (sfnp.float32, sfnp.int32, sfnp.ceil, math.ceil),
        # Floor
        (sfnp.float16, sfnp.int8, sfnp.floor, math.floor),
        (sfnp.float32, sfnp.int8, sfnp.floor, math.floor),
        (sfnp.float32, sfnp.int16, sfnp.floor, math.floor),
        (sfnp.float32, sfnp.int32, sfnp.floor, math.floor),
        # Trunc
        (sfnp.float16, sfnp.int8, sfnp.trunc, math.trunc),
        (sfnp.float32, sfnp.int8, sfnp.trunc, math.trunc),
        (sfnp.float32, sfnp.int16, sfnp.trunc, math.trunc),
        (sfnp.float32, sfnp.int32, sfnp.trunc, math.trunc),
    ],
)
def test_nearest_int_conversion(device, dtype, out_dtype, sfnp_func, ref_round_func):
    input = sfnp.device_array(device, [2, 3], dtype=dtype)
    sfnp.fill_randn(input)
    ref_rounded = [
        int(ref_round_func(n)) for n in sfnp.convert(input, dtype=sfnp.float32).items
    ]
    output = sfnp_func(input, dtype=out_dtype)
    assert output.dtype == out_dtype
    for ref, actual in zip(ref_rounded, output.items):
        assert ref == int(actual)


def test_elementwise_forms(device):
    # All elementwise ops use the same template expansion which enforces
    # certain common invariants. Here we test these on the multiply op,
    # relying on a parametric test for actual behavior.
    with pytest.raises(
        ValueError,
        match="Elementwise operators require at least one argument to be a device_array",
    ):
        sfnp.multiply(2, 2)

    ary = sfnp.device_array.for_host(device, [2, 3], dtype=sfnp.float32)
    with ary.map(discard=True) as m:
        m.fill(42.0)

    # Rhs scalar int accepted.
    result = sfnp.multiply(ary, 2)
    assert list(result.items) == [84.0] * 6

    # Rhs scalar float accepted.
    result = sfnp.multiply(ary, 2.0)
    assert list(result.items) == [84.0] * 6

    # Lhs scalar int accepted.
    result = sfnp.multiply(2, ary)
    assert list(result.items) == [84.0] * 6

    # Lhs scalar float accepted.
    result = sfnp.multiply(2.0, ary)
    assert list(result.items) == [84.0] * 6

    # Out.
    out = sfnp.device_array.for_host(device, [2, 3], dtype=sfnp.float32)
    sfnp.multiply(2.0, ary, out=out)
    assert list(out.items) == [84.0] * 6


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,promoted_dtype",
    [
        (sfnp.float32, sfnp.bfloat16, sfnp.float32),
        (sfnp.bfloat16, sfnp.float32, sfnp.float32),
        (sfnp.float32, sfnp.float16, sfnp.float32),
        (sfnp.float16, sfnp.float32, sfnp.float32),
        (sfnp.float32, sfnp.float64, sfnp.float64),
        (sfnp.float64, sfnp.float32, sfnp.float64),
        # Integer promotion.
        (sfnp.uint8, sfnp.uint16, sfnp.uint16),
        (sfnp.uint16, sfnp.uint32, sfnp.uint32),
        (sfnp.uint32, sfnp.uint64, sfnp.uint64),
        (sfnp.int8, sfnp.int16, sfnp.int16),
        (sfnp.int16, sfnp.int32, sfnp.int32),
        (sfnp.int32, sfnp.int64, sfnp.int64),
        # Signed/unsigned promotion.
        (sfnp.int8, sfnp.uint8, sfnp.int16),
        (sfnp.int16, sfnp.uint16, sfnp.int32),
        (sfnp.int32, sfnp.uint32, sfnp.int64),
        (sfnp.int8, sfnp.uint32, sfnp.int64),
    ],
)
def test_elementwise_promotion(device, lhs_dtype, rhs_dtype, promoted_dtype):
    # Tests that promotion infers an appropriate result type.
    lhs = sfnp.device_array.for_host(device, [2, 3], lhs_dtype)
    rhs = sfnp.device_array.for_host(device, [2, 3], rhs_dtype)
    result = sfnp.multiply(lhs, rhs)
    assert result.dtype == promoted_dtype


@pytest.mark.parametrize(
    "dtype,op,check_value",
    [
        # Add.
        (sfnp.int8, sfnp.add, 44.0),
        (sfnp.int16, sfnp.add, 44.0),
        (sfnp.int32, sfnp.add, 44.0),
        (sfnp.int64, sfnp.add, 44.0),
        (sfnp.uint8, sfnp.add, 44.0),
        (sfnp.uint16, sfnp.add, 44.0),
        (sfnp.uint32, sfnp.add, 44.0),
        (sfnp.uint64, sfnp.add, 44.0),
        (sfnp.bfloat16, sfnp.add, 44.0),
        (sfnp.float16, sfnp.add, 44.0),
        (sfnp.float32, sfnp.add, 44.0),
        (sfnp.float64, sfnp.add, 44.0),
        # Divide.
        (sfnp.int8, sfnp.divide, 21.0),
        (sfnp.int16, sfnp.divide, 21.0),
        (sfnp.int32, sfnp.divide, 21.0),
        (sfnp.int64, sfnp.divide, 21.0),
        (sfnp.uint8, sfnp.divide, 21.0),
        (sfnp.uint16, sfnp.divide, 21.0),
        (sfnp.uint32, sfnp.divide, 21.0),
        (sfnp.uint64, sfnp.divide, 21.0),
        (sfnp.bfloat16, sfnp.divide, 21.0),
        (sfnp.float16, sfnp.divide, 21.0),
        (sfnp.float32, sfnp.divide, 21.0),
        (sfnp.float64, sfnp.divide, 21.0),
        # Multiply.
        (sfnp.int8, sfnp.multiply, 84.0),
        (sfnp.int16, sfnp.multiply, 84.0),
        (sfnp.int32, sfnp.multiply, 84.0),
        (sfnp.int64, sfnp.multiply, 84.0),
        (sfnp.uint8, sfnp.multiply, 84.0),
        (sfnp.uint16, sfnp.multiply, 84.0),
        (sfnp.uint32, sfnp.multiply, 84.0),
        (sfnp.uint64, sfnp.multiply, 84.0),
        (sfnp.bfloat16, sfnp.multiply, 84.0),
        (sfnp.float16, sfnp.multiply, 84.0),
        (sfnp.float32, sfnp.multiply, 84.0),
        (sfnp.float64, sfnp.multiply, 84.0),
        # Subtract.
        (sfnp.int8, sfnp.subtract, 40.0),
        (sfnp.int16, sfnp.subtract, 40.0),
        (sfnp.int32, sfnp.subtract, 40.0),
        (sfnp.int64, sfnp.subtract, 40.0),
        (sfnp.uint8, sfnp.subtract, 40.0),
        (sfnp.uint16, sfnp.subtract, 40.0),
        (sfnp.uint32, sfnp.subtract, 40.0),
        (sfnp.uint64, sfnp.subtract, 40.0),
        (sfnp.bfloat16, sfnp.subtract, 40.0),
        (sfnp.float16, sfnp.subtract, 40.0),
        (sfnp.float32, sfnp.subtract, 40.0),
        (sfnp.float64, sfnp.subtract, 40.0),
    ],
)
def test_elementwise_array_correctness(device, dtype, op, check_value):
    lhs = sfnp.device_array.for_host(device, [2, 2], sfnp.int32)
    with lhs.map(discard=True) as m:
        m.fill(42)

    rhs = sfnp.device_array.for_host(device, [2], sfnp.int32)
    with rhs.map(discard=True) as m:
        m.fill(2)

    lhs = sfnp.convert(lhs, dtype=dtype)
    rhs = sfnp.convert(rhs, dtype=dtype)
    result = op(lhs, rhs)
    assert result.shape == [2, 2]
    result = sfnp.convert(result, dtype=sfnp.float32)
    items = list(result.items)
    assert items == [check_value] * 4


@pytest.mark.parametrize(
    "dtype",
    [
        sfnp.int8,
        sfnp.int16,
        sfnp.int32,
        sfnp.int64,
        sfnp.uint8,
        sfnp.uint16,
        sfnp.uint32,
        sfnp.uint64,
        sfnp.float32,
        sfnp.bfloat16,
        sfnp.float16,
        sfnp.float32,
        sfnp.float64,
    ],
)
def test_transpose(device, dtype):
    input = sfnp.device_array.for_host(device, [3, 2], sfnp.int32)
    input.items = [0, 1, 2, 3, 4, 5]
    input = sfnp.convert(input, dtype=dtype)
    permuted = sfnp.transpose(input, [1, 0])
    assert permuted.shape == [2, 3]
    items = list(sfnp.convert(permuted, dtype=sfnp.int32).items)
    assert items == [0, 2, 4, 1, 3, 5]

    out = sfnp.device_array.for_host(device, [2, 3], dtype)
    sfnp.transpose(input, [1, 0], out=out)
    items = list(sfnp.convert(permuted, dtype=sfnp.int32).items)
    assert items == [0, 2, 4, 1, 3, 5]
