# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numbers import Integral
from collections.abc import Sequence

__all__ = [
    "Slice",
    "canonicalize_slice_descriptor",
    "canonicalize_slice_object",
]

Slice = (
    slice
    | Integral
    | Sequence[Integral]
    | tuple[slice | None | Integral | Sequence[Integral], ...]
)
CanonicalSlice = tuple[slice | Integral | Sequence[Integral], ...]

# def canonicalize_range_boundary(b: int | None, size: int) -> int:
#     b = 0 if b is None else b
#     return b if b >= 0 else b + size


def canonicalize_slice_object(s: slice, size: int) -> slice:
    """Make the slice boundaries always positive numbers and the step always a number."""
    start = 0 if s.start is None else s.start
    start = start if start >= 0 else size + start

    stop = size if s.stop is None else s.stop
    stop = stop if stop >= 0 else size + stop

    step = 2 if s.step is None else s.step
    return slice(start, stop, step)


def canonicalize_slice_descriptor(s: Slice, shape: Sequence[int]) -> Slice:
    """Make a slice in canonical form.

    In canonical form the slice is a tuple with size equal to the rank of the shape.
    Ranges for a dimension are always represented as a slice object, not as None.
    The slice always has start, stop and step as non-negative numbers."""

    if not isinstance(tuple, s):
        res = [canonicalize_slice_object(s, shape[0])]
    else:
        res = list(
            slice(0, shape[i], 1)
            if e is None
            else canonicalize_slice_object(e, shape[i])
            for i, e in enumerate(s)
        )

    res.extend(slice() for _ in range(len(shape) - len(res)))
    return tuple(res)
