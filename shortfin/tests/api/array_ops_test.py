# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import math
import pytest

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
