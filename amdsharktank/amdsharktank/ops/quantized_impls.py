# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import functools
import math
import inspect
import torch
import warnings

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable
from torch import Tensor
from ._registry import *
from amdsharktank.types import (
    AnyTensor,
    DynamicFp4BlockQuantizer,
    DynamicScaledQuantizer,
    ReplicatedTensor,
    QuantizedTensor,
    BlockScaledFp4Layout,
    canonicalize_slice_descriptor,
    PlanarQuantizedTensor,
    PrimitiveTensor,
    QuantizedLayout,
    QuantizedTensor,
    QuantizerTensor,
    Slice,
    squeeze_slice,
    StaticFp4BlockQuantizer,
    StaticScaledQuantizer,
    TensorScaledLayout,
    unbox_tensor,
    UnnamedTensorName,
    unsqueeze_shape_for_slicing,
    unsqueeze_slice_like,
)
from amdsharktank.types.layout_utils import saturate_cast, unpack_uint8_to_fp4_e2m1
from amdsharktank.types.ocp_floats import (
    compute_fp4_block_scales,
    dynamic_quantize_to_fp4,
)
from amdsharktank.types.quantizers import (
    _fp4_block_quantize_tensor,
    pad_tensor_for_block_quantization,
)
from amdsharktank.ops.shape import cat_shape, normalize_negative_dim
from amdsharktank.utils import iterables_equal

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
            # torch.export doesn't play nicely with inspect
            if torch._dynamo.is_compiling():
                return f(*args, **kwargs)

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

        wrapper._layout_types = {}
        if layout_types:
            param_names = list(signature.parameters.keys())
            wrapper._layout_types.update(
                dict(zip(param_names[: len(layout_types)], layout_types))
            )
        if kw_layout_types:
            wrapper._layout_types.update(kw_layout_types)
        return wrapper

    return decorator


def verify_quantized_shape(actual: tuple[int, ...], expected: tuple[int, ...]):
    assert iterables_equal(
        actual, expected
    ), f"Quantization error, input and output shapes differ {expected} != {actual}"


@dequantize.override(dict, StaticScaledQuantizer)
def dequantize_planes_static_scaled_quantizer(
    input: dict[str, Tensor],
    quantizer: StaticScaledQuantizer,
    dtype: torch.dtype | None,
) -> Tensor:
    qs = input["qs"]
    if not isinstance(qs, (Tensor, PrimitiveTensor)):
        return NotImplemented
    qs = unbox_tensor(qs)

    return dequantize(
        PlanarQuantizedTensor(
            shape=qs.shape,
            layout=TensorScaledLayout(
                shape=qs.shape,
                d=quantizer._reciprocal_scale,
                qs=qs,
                m=quantizer.offset,
                dtype=dtype,
            ),
        )
    )


@dequantize.override(AllOfExprs(IsOfType(QuantizedTensor), BoolTypeExprConst(True)))
def dequantize_quantized_tensor(
    input: QuantizedTensor, quantizer: QuantizerTensor | None, dtype: torch.dtype | None
) -> Tensor:
    return input.unpack().dequant(dtype=dtype)


@quantize.override(Tensor, DynamicFp4BlockQuantizer)
def quantize_dynamic_fp4_block_quantizer(
    tensor: Tensor | PrimitiveTensor, quantizer: DynamicFp4BlockQuantizer, name: str
) -> QuantizedTensor:
    """Performs FP4 block quantization on tensor."""
    tensor = unbox_tensor(tensor)
    t_padded = pad_tensor_for_block_quantization(tensor, quantizer.block_size)

    # Compute scales per block
    orig_shape = list(t_padded.shape)
    num_blocks = orig_shape[-1] // quantizer.block_size
    blocked_shape = orig_shape[:-1] + [num_blocks, quantizer.block_size]
    packed_shape = orig_shape[:-1] + [num_blocks, quantizer.block_size // 2]
    values_blocked = t_padded.reshape(blocked_shape)

    if quantizer._use_amdsharktank_kernel:
        flattened = values_blocked.reshape(-1, quantizer.block_size).to(torch.float32)
        scales, packed_fp4_flat = dynamic_quantize_to_fp4(flattened)
        packed_fp4 = packed_fp4_flat.view(packed_shape)
        # Reshape scales to match the expected blocked dimensions
        scales_shape = orig_shape[:-1] + [num_blocks, 1]
        scales = scales.view(scales_shape)

        layout = BlockScaledFp4Layout(
            shape=list(tensor.shape),
            d=scales,
            qs=packed_fp4,
            block_size=quantizer.block_size,
            use_fe8m0_scale=quantizer.use_fe8m0_scale,
        )
        return PlanarQuantizedTensor(
            shape=list(tensor.shape),
            name=name,
            layout=layout,
        )
    block_max = torch.max(torch.abs(values_blocked), dim=-1, keepdim=False)[0]
    scales, _ = compute_fp4_block_scales(
        block_max, quantizer.use_fe8m0_scale, quantizer.dtype
    )

    res = _fp4_block_quantize_tensor(
        t=t_padded,
        scales=scales,
        block_size=quantizer.block_size,
        use_fe8m0_scale=quantizer.use_fe8m0_scale,
        name=name,
    )
    verify_quantized_shape(res.shape, tensor.shape)
    return res


@quantize.override(Tensor, DynamicScaledQuantizer)
def quantize_dynamic_scaled_quantizer(
    tensor: Tensor | PrimitiveTensor, quantizer: DynamicScaledQuantizer, name: str
) -> QuantizedTensor:
    tensor = unbox_tensor(tensor)
    dtype = quantizer._dtype
    amax = torch.max(torch.abs(tensor))
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        scale = finfo.max / amax.clamp(finfo.eps)
        reciprocal_scale = 1 / scale
        qs = saturate_cast(tensor * scale, quantizer.dtype, round_int=True)
    else:
        eps = 1e-6
        iinfo = torch.iinfo(dtype)
        scale = iinfo.max / amax.clamp(eps)
        reciprocal_scale = 1.0 / scale
        qs = saturate_cast(tensor * scale, quantizer.dtype, round_int=True)
    shape = list(tensor.shape)
    res = PlanarQuantizedTensor(
        shape=shape,
        name=name,
        layout=TensorScaledLayout(
            shape=shape,
            d=reciprocal_scale,
            qs=qs,
            dtype=tensor.dtype,  # Original dtype.
        ),
    )
    verify_quantized_shape(res.shape, tensor.shape)
    return res


@quantize.override(QuantizedTensor, QuantizerTensor)
def quantize_quantized(
    tensor: QuantizedTensor, quantizer: QuantizerTensor, name: str
) -> QuantizedTensor:
    """ "This has some additional heuristics for unpacking and rescaling."""
    warnings.warn(f"Requantizing already quantized tensor {tensor} to {quantizer}")
    raw_tensor = tensor.unpack().dequant()
    return quantize(raw_tensor, quantizer, name=name)


@quantize.override(Tensor, StaticFp4BlockQuantizer)
def quantize_static_fp4_block_quantizer(
    tensor: Tensor | PrimitiveTensor, quantizer: StaticFp4BlockQuantizer, name: str
) -> QuantizedTensor:
    """Performs FP4 block quantization on tensor using pre-computed scales."""
    tensor = unbox_tensor(tensor)
    res = _fp4_block_quantize_tensor(
        t=tensor,
        scales=quantizer.scales,
        block_size=quantizer._block_size,
        use_fe8m0_scale=quantizer._use_fe8m0_scale,
        name=name,
    )
    verify_quantized_shape(res.shape, tensor.shape)
    return res


@quantize.override(Tensor, StaticScaledQuantizer)
def quantize_static_scaled_quantizer(
    tensor: Tensor | PrimitiveTensor, quantizer: StaticScaledQuantizer, name: str
) -> QuantizedTensor:
    """Performs a quantizing transformation on tensor, returning a QuantizeTensor."""
    tensor = unbox_tensor(tensor)
    shape = list(tensor.shape)
    axis = quantizer._axis
    offset = quantizer._offset
    if axis is None:
        # Per tensor.
        if offset is None:
            # Changed to t/reciprocal because narrow float types are garbage
            qs = saturate_cast(
                tensor / quantizer._reciprocal_scale,
                dtype=quantizer.dtype,
                disable_saturate=quantizer._disable_saturate,
            )
        else:
            qs = saturate_cast(
                tensor / quantizer._reciprocal_scale + offset,
                dtype=quantizer.dtype,
                disable_saturate=quantizer._disable_saturate,
            )
        res = PlanarQuantizedTensor(
            shape=shape,
            name=name,
            layout=TensorScaledLayout(
                shape=shape,
                d=quantizer._reciprocal_scale,
                qs=qs,
                m=quantizer._offset,
                dtype=tensor.dtype,  # Original dtype.
            ),
        )
    else:
        # Expand the scale/reciprocal to correspond to the broadcast axis.
        scale = quantizer._scale
        reciprocal_scale = quantizer._reciprocal_scale
        offset = quantizer._offset
        assert axis >= 0 and axis < len(
            shape
        ), f"Per-axis scale {axis} out of bounds of shape {shape}"
        scale_shape = [1] * len(shape)
        scale_shape[axis] = scale.shape[0]
        broadcast_scale = scale.reshape(scale_shape)
        broadcast_reciprocal_scale = reciprocal_scale.reshape(scale_shape)
        if offset is None:
            broadcast_offset = None
            qs = saturate_cast(
                tensor * broadcast_scale,
                dtype=quantizer.dtype,
                disable_saturate=quantizer._disable_saturate,
            )
        else:
            broadcast_offset = offset.reshape(scale_shape)
            qs = saturate_cast(
                tensor * broadcast_scale + broadcast_offset,
                dtype=quantizer.dtype,
                disable_saturate=quantizer._disable_saturate,
            )
        res = PlanarQuantizedTensor(
            shape=shape,
            name=name,
            layout=TensorScaledLayout(
                shape=shape,
                d=broadcast_reciprocal_scale,
                qs=qs,
                m=broadcast_offset,
                dtype=tensor.dtype,  # Original dtype.
            ),
        )

    verify_quantized_shape(res.shape, tensor.shape)
    return res


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


@cat.override(AllOfType(PlanarQuantizedTensor))
def cat_BlockScaledFp4Layout(tensors: Sequence[PlanarQuantizedTensor], dim: int):
    if not all(issubclass(t.layout_type, BlockScaledFp4Layout) for t in tensors):
        return NotImplemented

    assert all(
        t.layout.block_size == tensors[0].layout.block_size for t in tensors
    ), "Concatenating tensors with layout BlockScaledFp4Layout with different block size is not supported"
    assert all(
        t.layout.use_fe8m0_scale == tensors[0].layout.use_fe8m0_scale for t in tensors
    ), "Concatenating tensors with layout BlockScaledFp4Layout with different scale dtypes is not supported."
    dim = normalize_negative_dim(tensors[0], dim)
    d = torch.cat([t.layout.d for t in tensors], dim)
    qs = torch.cat([t.layout.qs_bit_packed for t in tensors], dim)
    shape = cat_shape(*[t.shape for t in tensors], dim=dim)
    layout = BlockScaledFp4Layout(
        shape=shape,
        d=d,
        qs=qs,
        block_size=tensors[0].layout.block_size,
        use_fe8m0_scale=tensors[0].layout.use_fe8m0_scale,
    )
    return PlanarQuantizedTensor(
        shape=shape,
        layout=layout,
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

    block_scale_slice.append(slice(None))

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


@extract_slice.override(PlanarQuantizedTensor)
@quantized_tensor_layout_of_type(tensor=TensorScaledLayout)
def extract_slice_TensorScaledLayout(
    tensor: PlanarQuantizedTensor, key: Slice
) -> PlanarQuantizedTensor:
    planes = dict(tensor.layout.planes)
    planes["qs"] = extract_slice(planes["qs"], key)
    metadata = dict(tensor.layout.metadata)
    metadata["shape"] = tensor.shape
    return PlanarQuantizedTensor(
        shape=tensor.shape,
        layout=type(tensor.layout).create(
            shape=tensor.layout.shape, metadata=metadata, planes=planes
        ),
    )


@split.override(QuantizedTensor)
@quantized_tensor_layout_of_type(tensor=BlockScaledFp4Layout)
def split_BlockScaledFp4Layout(
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
        return split_BlockScaledFp4Layout(tensor, sections, dim)

    assert len(split_size_or_sections) > 0
    parts_range = [(0, split_size_or_sections[0])]
    for s in split_size_or_sections[1:]:
        parts_range.append((parts_range[-1][1], parts_range[-1][1] + s))
    assert parts_range[-1][1] == dim_size

    res = []
    for begin, end in parts_range:
        slice_ = tuple(
            slice(begin, end) if i == dim else slice(None) for i in range(dim + 1)
        )
        res.append(tensor[slice_])
    return tuple(res)


@to.override(PlanarQuantizedTensor)
def to_planar_quantized(
    tensor: PlanarQuantizedTensor, *args, **kwargs
) -> PrimitiveTensor:
    arg0 = args[0] if len(args) > 0 else None
    device_overload = ("device" in kwargs) or isinstance(arg0, (str, torch.device))
    assert device_overload, "Only transferring to another device is supported"

    def _transform(planes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(*args, **kwargs) for k, v in planes.items()}

    return tensor.transform_subtensors(_transform, copy_external_tensor_trait=False)


@unpack.override(PlanarQuantizedTensor)
def unpack_default(input: PlanarQuantizedTensor) -> QuantizedLayout:
    return input.layout


@unpack_qs.override(Tensor, BlockScaledFp4Layout)
def unpack_qs_block_scaled_fp4_layout(
    qs: Tensor | PrimitiveTensor, layout: BlockScaledFp4Layout
) -> AnyTensor:
    return unpack_uint8_to_fp4_e2m1(unbox_tensor(qs))


@unpack_to_qs.override(PlanarQuantizedTensor)
def unpack_to_qs_default(input: PlanarQuantizedTensor) -> Tensor:
    return unpack(input).qs
