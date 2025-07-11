# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import math
import inspect

from typing import Any, Callable
from torch import Tensor

from sharktank.types import (
    BlockScaledI4Layout,
    BlockScaledFp4Layout,
    canonicalize_slice_descriptor,
    PlanarQuantizedTensor,
    QuantizedTensor,
    QuantizedLayout,
    Slice,
    TensorScaledLayout,
)

from sharktank.ops.shape import normalize_negative_dim
from .signatures import *


def quantized_tensor_layout_of_type(
    *layout_types: tuple[QuantizedLayout | None],
    **kw_layout_types: dict[str, QuantizedLayout | None],
) -> Callable[..., Any]:
    """Decorator that check that the arguments have the expected QuantizedLayout.

    If the arguments have the expected layout call the function. If not, return NotImplemented.

    E.g.
    ```
    @my_fn.override(QuantizedTensor)
    @quantized_tensor_layout_of_type(a=BlockScaledFp4Layout, b=SuperBlockOffsetScaled_4_6_Layout)
    def my_fn_impl(a: QuantizedTensor, b: QuantizedTensor):
        ...
    ```

    """

    def decorator(f: Callable[..., Any]):
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind(*args, **kwargs)
            bound_layout_types = signature.bind(*layout_types, **kw_layout_types)
            for k, layout_type in bound_layout_types.arguments.items():
                if layout_type is None:
                    continue
                if signature.parameters[k].kind == inspect.Parameter.VAR_POSITIONAL:
                    if any(
                        not isinstance(arg.layout_type, l_type)
                        for l_type, arg in zip(
                            layout_type, bound_arguments.arguments[k]
                        )
                    ):
                        return NotImplemented
                if signature.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                    if any(
                        not isinstance(bound_arguments.arguments[k][name], l_type)
                        for name, l_type in layout_type.items()
                    ):
                        return NotImplemented
                if not isinstance(bound_arguments.arguments[k], layout_type):
                    return NotImplemented

            # All tensors have the expected layout, we can make the call.
            return f(*args, **kwargs)

        return wrapper

    return decorator


@extract_slice.override(QuantizedTensor)
def extract_slice_QuantizedTensor(tensor: QuantizedTensor, key: slice):
    unpacked = tensor.unpack()
    if isinstance(unpacked, BlockScaledI4Layout):
        mul = 2
        new_d = unpacked._d[key]
        new_qs = unpacked._qs[key]
        if unpacked.m is not None:
            new_m = unpacked.m[key]
        dims = new_qs.shape
        dims = dims[:-2] + (dims[-2] * dims[-1] * mul,)
        layout = BlockScaledI4Layout(shape=dims, d=new_d, qs=new_qs, m=new_m)
        return PlanarQuantizedTensor(shape=dims, layout=layout)
    elif isinstance(unpacked, TensorScaledLayout):
        d = unpacked._d
        qs = unpacked._qs[key]
        m = unpacked._m[key]
        shape = qs.shape
        layout = TensorScaledLayout(shape=shape, d=d, qs=qs, m=m)
        return PlanarQuantizedTensor(shape=shape, layout=layout)
    return NotImplemented


@extract_slice.override(QuantizedTensor)
@quantized_tensor_layout_of_type(tensor=BlockScaledFp4Layout)
def extract_slice_QuantizedTensor(tensor: QuantizedTensor, key: Slice):
    layout: BlockScaledFp4Layout = tensor.to_planar().layout
    slice_ = canonicalize_slice_descriptor(key, tensor.shape)
    assert all(
        isinstance(s, slice) and s.step == 1 for s in slice_
    ), "Slicing with integers like tensor[1, 2, [3, 4]] is not supported. Only ranges with step=1 are supported."
    block_shape = tuple(tensor.shape[i] // layout.d.shape[i] for i in len(tensor.shape))
    assert (
        math.prod(block_shape) == layout.block_size
    ), f"The block size {math.prod(block_shape)} derived from the layout shape does not match the block size {layout.block_size}"
    assert all(
        s >= 2 for s in block_shape
    ), f"Expected block shape with dimension sizes of at least 2 (due to packing), but got {block_shape}"
    assert all(
        tensor.shape[i] % block_shape == 0 for i in len(tensor.shape)
    ), "Only slicing at a block boundary is supported."


@split.override(QuantizedTensor)
@quantized_tensor_layout_of_type(tensor=BlockScaledFp4Layout)
def split_block_scaled_fp4(
    tensor: QuantizedTensor,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[QuantizedTensor, ...]:
    dim = normalize_negative_dim(tensor, dim)
    dim_size = tensor.shape[dim]
    if isinstance(split_size_or_sections, int):
        sections = [split_size_or_sections] * (dim_size // split_size_or_sections)
        reminder = dim_size % split_size_or_sections
        if reminder != 0:
            sections.append(reminder)
        return split_block_scaled_fp4(tensor, sections, dim)

    assert len(split_size_or_sections) > 0
    parts_range = [(0, split_size_or_sections[0])]
    for s in split_size_or_sections[1:]:
        parts_range.append((parts_range[-1][1], parts_range[-1][1] + s))
    assert parts_range[-1][1] == dim_size

    multi_dim_slice = [None] * len(parts_range)
    res = []
    for begin, end in parts_range:
        multi_dim_slice[dim] = slice(begin, end)
        res.append(tensor[multi_dim_slice])
        multi_dim_slice[dim] = None
    return tuple(res)
