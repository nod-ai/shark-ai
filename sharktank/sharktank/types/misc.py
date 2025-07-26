# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Sequence

__all__ = [
    "canonicalize_index",
    "canonicalize_slice_descriptor",
    "canonicalize_slice_object",
    "CanonicalSlice",
    "Slice",
    "squeeze_slice",
    "unsqueeze_shape_for_slicing",
    "unsqueeze_slice_like",
]

Slice = (
    slice | None | int | Sequence[int] | tuple[slice | None | int | Sequence[int], ...]
)
CanonicalSlice = tuple[slice | int | Sequence[int], ...]
"""In canonical form the slice is a tuple with size equal to the rank of the shape +
number of singleton dimensions to insert.
Ranges for a dimension are always represented as a slice object, and insertion of singleton dimensions as None.
The slice always has start, stop and step as non-negative numbers.
Indices are always positive.
"""


def canonicalize_slice_descriptor(s: Slice, shape: Sequence[int]) -> CanonicalSlice:
    """Make a slice in canonical form."""
    if not isinstance(s, tuple):
        s = (s,)

    slice_ = squeeze_slice(s)

    res = list(
        canonicalize_slice_object(e, shape[i])
        if isinstance(e, slice)
        else canonicalize_index(e, shape[i])
        for i, e in enumerate(slice_)
    )

    res.extend(slice(0, shape[i], 1) for i in range(len(res), len(shape)))
    return unsqueeze_slice_like(tuple(res), s)


def canonicalize_index(index: int | Sequence[int], size: int) -> int | Sequence[int]:
    """Make the index positive. size + index for size < 0."""

    if isinstance(index, int):
        if size < abs(index):
            raise IndexError(f"Index {index} out of bounds ({-size}, {size})")
        return index if index >= 0 else size + index

    return [canonicalize_index(i, size) for i in index]


def canonicalize_slice_object(s: slice, size: int) -> slice:
    """Make the slice boundaries always positive numbers and the step always a number.

    E.g.
    For size=3
    slice(None, None, None) -> slice(0, 3, 1)
    slice(-2, -1, 2) -> slice(1, 2, 2)
    """
    start = 0 if s.start is None else s.start
    start = canonicalize_index(start, size)

    stop = size if s.stop is None else s.stop
    stop = canonicalize_index(stop, size)

    step = 1 if s.step is None else s.step
    return slice(start, stop, step)


def squeeze_slice(s: Slice) -> Slice:
    """Remove Nones that represent insertion of a singleton dimensions.

    In slicing None represents an insertion of a dimension with size 1.

    E.g.
    (None, 1, None, slice(1), None) -> (1, slice(1))"""
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
