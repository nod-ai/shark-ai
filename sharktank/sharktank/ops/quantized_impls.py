# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import functools
import math
import inspect

from copy import deepcopy
from typing import Any, Callable
from torch import Tensor
from ._registry import *
from sharktank.types import (
    ReplicatedTensor,
    QuantizedTensor,
    BlockScaledFp4Layout,
    canonicalize_slice_descriptor,
    PlanarQuantizedTensor,
    QuantizedLayout,
    QuantizedTensor,
    Slice,
    squeeze_slice,
    unsqueeze_shape_for_slicing,
    unsqueeze_slice_like,
)
from sharktank.types.quantizers import QuantizerTensor
from sharktank.ops.shape import normalize_negative_dim

from .signatures import *

import iree.turbine.ops.iree


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
            bound_layout_types = signature.bind_partial(
                *layout_types, **kw_layout_types
            )
            for k, layout_type in bound_layout_types.arguments.items():
                if layout_type is None:
                    continue
                if signature.parameters[k].kind == inspect.Parameter.VAR_POSITIONAL:
                    if any(
                        not isinstance(arg.to_planar().layout, l_type)
                        for l_type, arg in zip(
                            layout_type, bound_arguments.arguments[k]
                        )
                    ):
                        return NotImplemented
                if signature.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                    if any(
                        not isinstance(
                            bound_arguments.arguments[k][name].to_planar().layout,
                            l_type,
                        )
                        for name, l_type in layout_type.items()
                    ):
                        return NotImplemented
                if not isinstance(
                    bound_arguments.arguments[k].to_planar().layout, layout_type
                ):
                    return NotImplemented

            # All tensors have the expected layout, we can make the call.
            return f(*args, **kwargs)

        return wrapper

    return decorator


@replicate.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def replicate_quantized(
    tensor: QuantizedTensor | QuantizerTensor, *, count: int, devices: tuple[int, ...]
) -> ReplicatedTensor:
    assert count == len(devices)
    return ReplicatedTensor(ts=tensor, shard_count=count, devices=devices)


@transfer_to_logical_device.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def transfer_to_logical_device_planar_quantized_tensor(
    tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.transfer_to_logical_device, tensor, ordinal
    )


@barrier_on_logical_device.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def barrier_on_logical_device_planar_quantized_tensor(
    tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.barrier_on_logical_device, tensor, ordinal
    )


def transfer_or_barrier(
    operation: Callable, tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    def operation_transform(globals: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: operation(f"{ordinal}", v) for k, v in globals.items()}

    return tensor.transform_subtensors(
        operation_transform, copy_external_tensor_trait=False
    )


@extract_slice.override(PlanarQuantizedTensor)
@quantized_tensor_layout_of_type(tensor=BlockScaledFp4Layout)
def extract_slice_BlockScaledFp4Layout(tensor: PlanarQuantizedTensor, key: Slice):
    layout: BlockScaledFp4Layout = tensor.layout
    slice_ = canonicalize_slice_descriptor(squeeze_slice(key), tensor.shape)
    assert all(
        isinstance(s, slice) for s in slice_
    ), "Slicing with integers like tensor[1, 2, [3, 4]] is not supported. Only ranges are supported."
    block_shape = tuple(
        tensor.shape[i] // layout.d.shape[i] for i in range(len(tensor.shape))
    )
    assert (
        math.prod(block_shape) == layout.block_size
    ), f"The block size {math.prod(block_shape)} derived from the layout shape does not match the block size {layout.block_size}"
    assert all(
        s >= 2 for s in block_shape if s != 1
    ), f"Expected block shape with non-singleton dimension sizes of at least 2 (due to packing), but got {block_shape}"
    assert all(
        slice_[i].step == 1 for i in range(len(slice_))
    ), f"Slicing with a step other than 1 is not supported."
    assert all(
        slice_[i].start % block_shape[i] == 0 and slice_[i].stop % block_shape[i] == 0
        for i in range(len(slice_))
    ), "Only slicing at a block boundary is supported."

    slice_ = [slice(s.start // b, s.stop // b) for s, b in zip(slice_, block_shape)]
    block_scale_slice = deepcopy(slice_)

    slice_ = list(slice_)
    # One more dimension for indexing in the block.
    slice_.append(slice(None))

    # TODO: Remove assert and enable this when BlockScaledFp4Layout aligns with BlockScaledLayout.
    assert len(layout.d.shape) + 1 == len(layout.qs_bit_packed.shape)
    # block_scale_slice.append(slice(None))

    # Reintroduce singleton dimension inserts.
    slice_ = unsqueeze_slice_like(tuple(slice_), like=key)
    block_scale_slice = unsqueeze_slice_like(tuple(block_scale_slice), like=key)

    result_qs = layout.qs_bit_packed[slice_]
    result_d = layout.d[block_scale_slice]
    result_shape = tuple(
        x * y
        for x, y in zip(
            result_qs.shape[:-1],
            unsqueeze_shape_for_slicing(block_shape, key),
            strict=True,
        )
    )
    result_layout = BlockScaledFp4Layout(
        shape=result_shape,
        d=result_d,
        qs=result_qs,
        block_size=layout.block_size,
        use_fe8m0_scale=layout.use_fe8m0_scale,
    )
    return PlanarQuantizedTensor(
        shape=result_shape,
        layout=result_layout,
    )
