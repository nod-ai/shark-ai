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


def test_storage_constructor(lsys, device):
    async def main():
        s = sfnp.storage.allocate_host(device, 8)
        s.fill(b"\0\1\2\3")
        await device
        ary = sfnp.device_array(s, [2, 4], sfnp.uint8)
        assert ary.dtype == sfnp.uint8
        assert ary.shape == [2, 4]
        assert list(ary.items) == [0, 1, 2, 3, 0, 1, 2, 3]
        assert ary.device == device
        assert ary.storage == s

    lsys.run(main())


def test_device_constructor(lsys, device):
    async def main():
        ary = sfnp.device_array(device, [2, 4], sfnp.uint8)
        ary.storage.fill(b"\0\1\2\3")
        await device
        assert ary.dtype == sfnp.uint8
        assert ary.shape == [2, 4]
        assert list(ary.items) == [0, 1, 2, 3, 0, 1, 2, 3]
        assert ary.device == device

    lsys.run(main())


def test_fill_copy_from_for_transfer(lsys, device):
    async def main():
        src = sfnp.device_array(device, [2, 4], sfnp.uint8)
        src.fill(b"\0\1\2\3")
        dst = src.for_transfer()
        dst.copy_from(src)
        await device
        assert list(dst.items) == [0, 1, 2, 3, 0, 1, 2, 3]

    lsys.run(main())


def test_fill_copy_to_for_transfer(lsys, device):
    async def main():
        src = sfnp.device_array(device, [2, 4], sfnp.uint8)
        src.fill(b"\0\1\2\3")
        dst = src.for_transfer()
        src.copy_to(dst)
        await device
        assert list(dst.items) == [0, 1, 2, 3, 0, 1, 2, 3]

    lsys.run(main())


def test_shape_overflow(lsys, device):
    async def main():
        s = sfnp.storage.allocate_host(device, 4)
        _ = sfnp.device_array(s, [2, 4], sfnp.uint8)

    with pytest.raises(
        ValueError, match="Array storage requires at least 8 bytes but has only 4"
    ):
        lsys.run(main())


@pytest.mark.parametrize(
    "dtype,code,py_value,expected_str",
    [
        (sfnp.int8, "b", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sfnp.int16, "h", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (sfnp.int32, "i", 42, "{{42, 42, 42, 42},\n {42, 42, 42, 42}}"),
        (
            sfnp.float32,
            "f",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
        (
            sfnp.float64,
            "d",
            42.0,
            "{{ 42.,  42.,  42.,  42.},\n { 42.,  42.,  42.,  42.}}",
        ),
    ],
)
def test_xtensor_types(fiber, dtype, code, py_value, expected_str):
    ary = sfnp.device_array.for_host(fiber.device(0), [2, 4], dtype)
    with ary.map(discard=True) as m:
        m.fill(py_value)
    s = str(ary)
    print("__str__ =", s)
    assert expected_str == s, f"Expected '{expected_str}' == '{s}'"
    r = repr(ary)
    print("__repr__ =", r)
    assert expected_str in r, f"Expected '{expected_str}' in '{r}'"


@pytest.mark.parametrize(
    "dtype,value,",
    [
        (sfnp.int8, 42),
        (sfnp.int16, 42),
        (sfnp.int32, 42),
        (sfnp.int64, 42),
        (sfnp.float16, 42),
        (sfnp.float32, 42.0),
        (sfnp.float64, 42.0),
    ],
)
def test_items(fiber, dtype, value):
    ary = sfnp.device_array.for_host(fiber.device(0), [2, 4], dtype)
    ary.items = [value] * 8
    readback = ary.items.tolist()
    assert readback == [value] * 8


@pytest.mark.parametrize(
    "dtype,value,",
    [
        (sfnp.int8, 42),
        (sfnp.int16, 42),
        (sfnp.int32, 42),
        (sfnp.int64, 42),
        (sfnp.float32, 42.0),
        (sfnp.float64, 42.0),
    ],
)
def test_typed_mapping(fiber, dtype, value):
    ary = sfnp.device_array.for_host(fiber.device(0), [2, 4], dtype)
    with ary.map(discard=True) as m:
        m.fill(value)
    readback = ary.items.tolist()
    assert readback == [value] * 8

    # Map as read/write and validate.
    with ary.map(read=True, write=True) as m:
        new_values = m.items.tolist()
        for i in range(len(new_values)):
            new_values[i] += 1
        m.items = new_values

    readback = ary.items.tolist()
    assert readback == [value + 1] * 8


@pytest.mark.parametrize(
    "keys,expected",
    [
        # Simple indexing
        ([0, 0], [0]),
        # Row indexing
        ([0], [0, 1, 2, 3]),
        # Sliced indexing
        ([1, slice(2, 4)], [2, 3]),
        ([slice(1, 2), slice(2, 4)], [2, 3]),
    ],
)
def test_view(device, keys, expected):
    src = sfnp.device_array(device, [4, 4], sfnp.uint8)
    with src.map(discard=True) as m:
        m.fill(b"\0\1\2\3")
    view = src.view(*keys)
    assert list(view.items) == expected


def test_view_nd(device):
    shape = [4, 16, 128]
    data = [i for i in range(math.prod(shape))]
    src = sfnp.device_array(device, [4, 16, 128], dtype=sfnp.uint32)
    src.items = data

    # Validate left justified indexing into the first dimension.
    for i in range(4):
        v = src.view(i)
        v_items = v.items.tolist()
        assert len(v_items) == 2048
        assert v_items[0] == i * 2048
        assert v_items[-1] == (i + 1) * 2048 - 1
    for i in range(16):
        v = src.view(1, i)
        v_items = v.items.tolist()
        assert len(v_items) == 128
        assert v_items[0] == 2048 + i * 128
        assert v_items[-1] == 2048 + (i + 1) * 128 - 1
    for i in range(128):
        v = src.view(1, 1, 1)
        v_items = v.items.tolist()
        assert len(v_items) == 1
        assert v_items[0] == 2177

    # Validate span.
    for i in range(16):
        v = src.view(slice(2, 4))
        v_items = v.items.tolist()
        assert len(v_items) == 4096
        assert v_items[0] == 4096
        assert v_items[-1] == 8191


def test_view_unsupported(lsys, device):
    async def main():
        src = sfnp.device_array(device, [4, 4], sfnp.uint8)
        src.fill(b"\0\1\2\3")

        with pytest.raises(
            ValueError,
            match="Cannot create a view with dimensions following a spanning dim",
        ):
            view = src.view(slice(0, 2), 1)
            await device

    lsys.run(main())
