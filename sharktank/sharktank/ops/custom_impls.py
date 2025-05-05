# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from collections.abc import Iterable
from typing import Union
from torch import Tensor


from sharktank.kernels import (
    einsum_2args_q4,
    mmt_block_scaled_offset_q4_unsigned,
    mmt_block_scaled_q8,
    mmt_super_block_scaled_offset_q4_unsigned,
    bitcast_to_complex,
    bitcast_to_real,
)

from sharktank.types import (
    BlockScaledLayout,
    BlockScaledI4Layout,
    PrimitiveTensor,
    QuantizedTensor,
    SuperBlockOffsetScaled_4_6_Layout,
)

from sharktank.types.tensors import AnyTensor, unbox_tensor
from .signatures import *
from ._registry import NotOfType


# Fused FP matmul.
# Disabled: See https://github.com/nod-ai/shark-ai/issues/44
# @matmul.override(Tensor, Tensor)
# def matmul_mmtfp_tensor_tensor(lhs, rhs, *, transpose_rhs: bool):
#     lhs = unbox_tensor(lhs)
#     rhs = unbox_tensor(rhs)
#     # We only accelerate matmuls with transposed RHS set up for inference
#     # ... like civilized people.
#     if not transpose_rhs:
#         return NotImplemented
#     if len(lhs.shape) > 3:
#         # Only 2d or 3d batch matmuls currently supported.
#         return NotImplemented
#     return mmtfp(lhs, rhs)


# Einsum


@einsum_2args.override(Tensor, QuantizedTensor)
def einsum_2args_QuantizedTensor(input0, input1, einsum_str):
    unpacked = input1.unpack()
    layout = input1.layout_type
    if not isinstance(unpacked, BlockScaledI4Layout):
        return NotImplemented
    return einsum_2args_q4(input0, unpacked.d, unpacked._qs, unpacked.m, einsum_str)


# Quantized Matmul


@matmul.override(Tensor, QuantizedTensor)
def matmul_generic_tensor_block_scaled(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic fallback kernel for block scaled layouts.

    This will unpack and operate generically on planar scales/blocks vs a packed
    struct. This may be fine for certain platforms but there is micro-optimization
    potential if specializing further to the packed layout.
    """
    lhs = unbox_tensor(lhs)
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not BlockScaledLayout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    assert rhs_unpacked.m is None, "NYI: Q8 block scaled with offset"
    return mmt_block_scaled_q8(lhs, rhs_unpacked.d, rhs_unpacked.qs)


@matmul.override(Tensor, QuantizedTensor)
def matmul_generic_tensor_block_scaled_i4(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic fallback kernel for an unsigned, block scaled Q4."""
    lhs = unbox_tensor(lhs)
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not BlockScaledI4Layout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    assert rhs_unpacked.m is not None, "NYI: Q4 without offset not"
    assert not rhs_unpacked.signed, "NYI: Q4 signed"
    return mmt_block_scaled_offset_q4_unsigned(
        a=lhs, d=rhs_unpacked.d, qs=rhs_unpacked.qs_bit_packed, m=rhs_unpacked.m
    )


@matmul.override(Tensor, QuantizedTensor)
def matmul_generic_tensor_super_block_offset_scaled_4_6_i4(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    lhs = unbox_tensor(lhs)
    if not transpose_rhs:
        return NotImplemented
    layout = rhs.layout_type
    if layout is not SuperBlockOffsetScaled_4_6_Layout:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    sb_scales_hi, sb_scales_low = rhs_unpacked.sb_scales_bit_packed
    sb_mins_hi, sb_mins_low = rhs_unpacked.sb_mins_bit_packed
    return mmt_super_block_scaled_offset_q4_unsigned(
        lhs,
        rhs_unpacked.d,
        rhs_unpacked.dmin,
        sb_scales_hi,
        sb_scales_low,
        sb_mins_hi,
        sb_mins_low,
        rhs_unpacked.qs_bit_packed,
    )


@sum.override(NotOfType(AnyTensor))
def sum_iterable(
    input: Iterable,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
    *,
    dtype,
):
    """Sum over an iterable of tensors."""
    assert isinstance(input, Iterable), "argument must be an iterable"
    if dim is not None:
        raise NotImplementedError("dim is not supported for iterable sum")
    if keepdim:
        raise NotImplementedError("keepdim is not supported for iterable sum")
    if dtype is not None:
        raise NotImplementedError("dtype is not supported for iterable sum")
    return __builtins__["sum"](input)


@view_as_complex.override(Union[Tensor, PrimitiveTensor])
def view_as_complex(t):
    t = unbox_tensor(t)
    return bitcast_to_complex(t)


@view_as_real.override(Union[Tensor, PrimitiveTensor])
def view_as_real(t):
    t = unbox_tensor(t)
    return bitcast_to_real(t)
