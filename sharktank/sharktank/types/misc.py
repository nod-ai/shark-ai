# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numbers import Integral
from collections.abc import Sequence

__all__ = [
    "canonicalize_slice_descriptor",
    "canonicalize_slice_object",
    "Slice",
    "squeeze_slice",
    "unsqueeze_shape_for_slicing",
    "unsqueeze_slice_like",
]

Slice = (
    slice
    | Integral
    | Sequence[Integral]
    | tuple[slice | None | Integral | Sequence[Integral], ...]
)
CanonicalSlice = tuple[slice | Integral | Sequence[Integral], ...]
"""In canonical form the slice is a tuple with size equal to the rank of the shape +
number of singleton dimensions to insert.
Ranges for a dimension are always represented as a slice object, and insertion of singleton dimensions as None.
The slice always has start, stop and step as non-negative numbers.
"""

# def canonicalize_range_boundary(b: int | None, size: int) -> int:
#     b = 0 if b is None else b
#     return b if b >= 0 else b + size


def canonicalize_slice_descriptor(s: Slice, shape: Sequence[int]) -> CanonicalSlice:
    """Make a slice in canonical form."""

    slice_ = squeeze_slice(s)

    if not isinstance(s, tuple):
        res = [canonicalize_slice_object(slice_, shape[0])]
    else:
        res = list(canonicalize_slice_object(e, shape[i]) for i, e in enumerate(slice_))

    res.extend(slice(0, shape[i], 1) for i in range(len(res), len(shape)))
    return unsqueeze_slice_like(tuple(res), s)


def canonicalize_slice_object(s: slice, size: int) -> slice:
    """Make the slice boundaries always positive numbers and the step always a number."""
    start = 0 if s.start is None else s.start
    start = start if start >= 0 else size + start

    stop = size if s.stop is None else s.stop
    stop = stop if stop >= 0 else size + stop

    step = 1 if s.step is None else s.step
    return slice(start, stop, step)


def squeeze_slice(s: Slice) -> Slice:
    """Remove Nones that represent insertion of a singleton dimensions."""
    if not isinstance(s, tuple):
        s = (s,)

    return tuple(e for e in s if e is not None)


def unsqueeze_shape_for_slicing(shape: Sequence[int], s: Slice) -> Sequence[int]:
    """Insert singleton dimensions for None dimension slice.

    E.g.
    ```
    unsqueeze_shape_for_slicing(shape=[2, 3, 4], s=(None, slice(), None))
    ```
    results in
    ```
    [1, 2, 1, 3, 4]
    ```
    """

    if not isinstance(s, tuple):
        s = (s,)

    res = []
    slice_idx = 0
    for dim in shape:
        while slice_idx < len(s) and s[slice_idx] is None:
            res.append(1)
            slice_idx += 1
        res.append(dim)
        slice_idx += 1
    return res


def unsqueeze_slice_like(s: Slice, like: Slice) -> Slice:
    """Insert Nones that represent insertion of a singleton dimensions."""
    if not isinstance(s, tuple):
        s = (s,)
    if not isinstance(like, tuple):
        like = (like,)

    res = []
    like_idx = 0
    for e in s:
        while like_idx < len(like) and like[like_idx] is None:
            res.append(None)
            like_idx += 1
        res.append(e)
        like_idx += 1
    return tuple(res)
