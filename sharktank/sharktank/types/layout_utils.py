# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from .ocp_floats import _FP4_MIN_INDEX, _FP4_MAX_INDEX


__all__ = [
    "debug_map_tensor_as_hex_string",
    "interleave_linear_i4_block",
    "linearize_interleaved_i4_block",
    "pack_fp4_e2m1_to_uint8",
    "pack_nibbles",
    "promote_linear_i2_block_to_i8",
    "promote_linear_i4_block_to_i8",
    "promote_linear_i6_block_to_i8",
    "saturate_cast",
    "unpack_nibbles",
    "unpack_uint8_to_fp4_e2m1",
]


def linearize_interleaved_i4_block(i8_data: torch.Tensor) -> torch.Tensor:
    """De-interleaves a tensor with an i4 block of data as its innermost dim.

    Given 4bit data of the form:
        0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7
    Converts to a linear form:
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe

    Such interleaved data often is a natural form for broadcasting direct from
    tensors of low and high nibbles to a larger bit-width, so it shows up a lot.
    The linearized version can be more useful for algorithms that are operating
    on a packed block directly or that prefer such register layouts.
    """
    i8_data = _view_uint8_tensor(i8_data)
    assert i8_data.dtype == torch.uint8, f"Expected uint8. Got {i8_data.dtype}"
    low_nibbles = i8_data & 0xF
    high_nibbles = i8_data >> 4
    low_even = low_nibbles[..., ::2]
    low_odd = low_nibbles[..., 1::2]
    high_even = high_nibbles[..., ::2]
    high_odd = high_nibbles[..., 1::2]
    t1 = (low_odd << 4) | low_even
    t2 = (high_odd << 4) | high_even
    linear = torch.cat([t1, t2], dim=-1)
    return linear


def interleave_linear_i4_block(i8_data: torch.Tensor) -> torch.Tensor:
    """Inverse of linearize_interleaved_i4_block."""
    i8_data = _view_uint8_tensor(i8_data)
    t1, t2 = torch.tensor_split(i8_data, 2, dim=-1)
    assert t1.size(-1) == t2.size(
        -1
    ), "interleave_linear_i4_block: must have even inner-most dim"
    low_even = t1 & 0xF
    low_odd = t1 >> 4
    high_even = t2 & 0xF
    high_odd = t2 >> 4
    i0 = (high_even << 4) | low_even
    i1 = (high_odd << 4) | low_odd
    i0 = i0.unsqueeze(-1)
    i1 = i1.unsqueeze(-1)
    stacked = torch.cat([i0, i1], dim=-1)
    interleaved = stacked.flatten(start_dim=-2)
    return interleaved


def unpack_nibbles(packed_data: torch.Tensor, *, signed: bool = False) -> torch.Tensor:
    """Unpack 4-bit values from uint8 tensor.

    Args:
        packed_data: Tensor of uint8 values containing packed 4-bit values
        signed: If True, treat nibbles as signed 4-bit integers

    Returns:
        Unpacked tensor with shape [..., 2*N] containing individual 4-bit values
    """
    packed_data = _view_uint8_tensor(packed_data)

    # Validate tensor type is uint8
    if packed_data.dtype != torch.uint8:
        raise TypeError(f"Packed data tensor must be uint8, got {packed_data.dtype}")

    if signed:
        # For signed i4 quantities, we have to manipulate the values as
        # right shifts from the high order nibble in order for sign extension
        # to function.
        low = (packed_data << 4).view(torch.int8) >> 4
        high = packed_data.view(torch.int8) >> 4
    else:
        low = packed_data & 0xF
        high = packed_data >> 4

    # Interleave back to original order
    low = low.unsqueeze(-1)
    high = high.unsqueeze(-1)
    stacked = torch.cat([low, high], dim=-1)
    flat = stacked.flatten(start_dim=-2)
    return flat


def pack_nibbles(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit values into uint8 tensor.

    Args:
        low: Tensor of lower nibbles (4-bit values)
        high: Tensor of upper nibbles (4-bit values)

    Returns:
        Packed uint8 tensor with shape [..., N//2]
    """
    low = _view_uint8_tensor(low)
    high = _view_uint8_tensor(high)

    # Validate tensor types are uint8
    if low.dtype != torch.uint8:
        raise TypeError(f"Low nibble tensor must be uint8, got {low.dtype}")
    if high.dtype != torch.uint8:
        raise TypeError(f"High nibble tensor must be uint8, got {high.dtype}")

    # Validate nibble value ranges - using torch._check for export compatibility
    torch._check(
        low.max().item() <= _FP4_MAX_INDEX,
        f"Low nibble values must be in range [{_FP4_MIN_INDEX}, {_FP4_MAX_INDEX}].",
    )
    torch._check(
        high.max().item() <= _FP4_MAX_INDEX,
        f"High nibble values must be in range [{_FP4_MIN_INDEX}, {_FP4_MAX_INDEX}].",
    )

    return low | (high << 4)


def promote_linear_i4_block_to_i8(
    linear_i4_data: torch.Tensor, *, signed: bool = False
) -> torch.Tensor:
    """Promote a linear i4 blocked tensor to i8."""
    return unpack_nibbles(linear_i4_data, signed=signed)


def promote_linear_i2_block_to_i8(linear_i2_data: torch.Tensor) -> torch.Tensor:
    """Promote a linear i4 blocked tensor to i8."""
    linear_i2_data = _view_uint8_tensor(linear_i2_data)
    assert linear_i2_data.dtype == torch.uint8, "NYI: Signed i2 promote to i8"
    d0 = linear_i2_data & 0x3
    d1 = (linear_i2_data >> 2) & 0x3
    d2 = (linear_i2_data >> 4) & 0x3
    d3 = (linear_i2_data >> 6) & 0x3
    stacked = torch.cat(
        [d0.unsqueeze(-1), d1.unsqueeze(-1), d2.unsqueeze(-1), d3.unsqueeze(-1)], dim=-1
    )
    flat = stacked.flatten(start_dim=-2)
    return flat


def promote_linear_i6_block_to_i8(
    i6_data_high: torch.Tensor, i6_data_low: torch.Tensor
) -> torch.Tensor:
    """Combines a 4 bit and 2 bit tensor into i8 values."""
    i4_data_low = promote_linear_i4_block_to_i8(i6_data_low)
    i2_data_high = promote_linear_i2_block_to_i8(i6_data_high)
    assert (
        i4_data_low.shape == i2_data_high.shape
    ), f"i4 low/high tensors should have the same shape ({i4_data_low.shape} vs {i2_data_high.shape})"
    return i4_data_low | (i2_data_high << 4)


def debug_map_tensor_as_hex_string(data: torch.Tensor) -> list:
    """Debug helper to print contents of a tensor mapped via hex().

    Returns a list with the same structure as the tensor but with all elements
    replaced with a hexadecimal string representation. Useful for debugging
    transformations on binary tensors.
    """

    def mapelt(x):
        if isinstance(x, list):
            return [mapelt(y) for y in x]
        return hex(x)

    return mapelt(data.tolist())


def _view_uint8_tensor(data: torch.Tensor) -> torch.Tensor:
    """Views an int8/uint8 tensor as uint8.

    Asserts if any other dtype.

    This helper is for performing raw bitwise manipulations on sub-byte values.
    If doing arithmetic bitwise, you will want to use a signed tensor and
    appropriate operations to manage sign extension.
    """
    dtype = data.dtype
    if dtype == torch.uint8:
        return data
    elif dtype == torch.int8:
        return data.view(torch.uint8)
    else:
        raise AssertionError(f"Expected tensor to by uint8 or int8. Got: {dtype}")


def saturate_cast(
    t: torch.Tensor,
    dtype: torch.dtype,
    round_int: bool = True,
    disable_saturate: bool = False,
) -> torch.Tensor:
    """Does a saturating cast to the given dtype. For floating point
    values, this is a simple cast except for fp8 which is saturated.
    For integer types, it will saturate to the min/max range.
    An argument disable_saturate= is provided to allow
    saturation to be disabled by flag without changing caller code. This is
    needed if (for example, trying to saturate a high precision integer
    type like int32) with a low precision tensor.
    """
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        isfp8 = finfo.bits == 8
        if isfp8 and not disable_saturate:
            t = t.clamp(finfo.min, finfo.max)
        return t.to(dtype=dtype)

    iinfo = torch.iinfo(dtype)
    if round_int:
        t = torch.round(t)
    if not disable_saturate:
        t = t.clamp(iinfo.min, iinfo.max)
    return t.to(dtype=dtype)


def pack_fp4_e2m1_to_uint8(fp4_values: torch.Tensor) -> torch.Tensor:
    """Pack pairs of FP4 E2M1 values into uint8 tensors.

    Args:
        fp4_values: Tensor of shape [..., N] where N is even, containing
                   4-bit values as uint8 (0-15 range)

    Returns:
        Packed tensor of shape [..., N//2] with 2 FP4 values per byte
    """
    if fp4_values.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for FP4 packing, got {fp4_values.shape[-1]}. "
            f"Ensure input tensor has an even number of FP4 values in the last dimension."
        )

    fp4_values = _view_uint8_tensor(fp4_values)

    # Validate tensor type is uint8
    if fp4_values.dtype != torch.uint8:
        raise TypeError(f"FP4 values tensor must be uint8, got {fp4_values.dtype}")

    # Validate FP4 value range
    torch._check(
        fp4_values.max().item() <= _FP4_MAX_INDEX,
        f"FP4 values must be in range [{_FP4_MIN_INDEX}, {_FP4_MAX_INDEX}]. "
        f"Use float32_to_fp4_e2m1() to convert float values to FP4 indices first.",
    )

    # Split into low and high nibbles
    low_nibbles = fp4_values[..., ::2]  # Even indices (0, 2, 4, ...)
    high_nibbles = fp4_values[..., 1::2]  # Odd indices (1, 3, 5, ...)

    return pack_nibbles(low_nibbles, high_nibbles)


def unpack_uint8_to_fp4_e2m1(packed_data: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 data into pairs of FP4 E2M1 values.

    Args:
        packed_data: Tensor of uint8 values with packed FP4 pairs

    Returns:
        Unpacked tensor with shape [..., 2*N] storing FP4 values in uint8s
    """
    return unpack_nibbles(packed_data, signed=False)
