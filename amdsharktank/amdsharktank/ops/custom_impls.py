# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from collections.abc import Iterable
from typing import Union
from torch import Tensor
import torch


from amdsharktank.kernels import (
    einsum_2args_q4,
    iree_mxfp4_bmm,
    mmt_block_scaled_offset_q4_unsigned,
    mmt_block_scaled_q8,
    mmt_super_block_scaled_offset_q4_unsigned,
    bitcast_to_complex,
    bitcast_to_real,
)

from amdsharktank.kernels.gemm_fp4_asm import (
    asm_fp4_gemm,
)
from amdsharktank.kernels.wave.mxfp4_gemm import wave_mxfp4_bmm

from amdsharktank.types import (
    BlockScaledLayout,
    BlockScaledI4Layout,
    BlockScaledFp4Layout,
    PrimitiveTensor,
    QuantizedTensor,
    SuperBlockOffsetScaled_4_6_Layout,
)
from amdsharktank.types.quantizers import DynamicFp4BlockQuantizer

from amdsharktank.types.tensors import AnyTensor, unbox_tensor
from amdsharktank.types.ocp_floats import convert_fp4_scales_to_float
from .signatures import *
from ._registry import AllNotOfType


# Fused FP matmul.
# Disabled: See https://github.com/nod-ai/amdshark-ai/issues/44
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


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank")
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


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank")
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


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank.iree")
def matmul_generic_tensor_block_scaled_fp4_iree(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic kernel for FP4 E2M1 block scaled layouts."""

    if rhs.layout_type is not BlockScaledFp4Layout:
        return NotImplemented

    lhs = unbox_tensor(lhs)
    if not transpose_rhs:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    quantizer = DynamicFp4BlockQuantizer(
        block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
    )
    lhs_quantized = quantizer.quantize(lhs)
    lhs_unpacked = lhs_quantized.unpack()
    return iree_mxfp4_bmm(
        lhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        lhs_unpacked.d.squeeze(-1),
        rhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        rhs_unpacked.d.squeeze(-1),
    )


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank.wave")
def matmul_generic_tensor_block_scaled_fp4_wave(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic kernel for FP4 E2M1 block scaled layouts."""

    if rhs.layout_type is not BlockScaledFp4Layout:
        return NotImplemented

    if not torch.compiler.is_compiling():
        lhs = unbox_tensor(lhs)
        rhs = unbox_tensor(rhs)
        return matmul(lhs, rhs, transpose_rhs=transpose_rhs)

    lhs = unbox_tensor(lhs)
    if not transpose_rhs:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    quantizer = DynamicFp4BlockQuantizer(
        block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
    )
    lhs_quantized = quantizer.quantize(lhs)
    lhs_unpacked = lhs_quantized.unpack()
    output = torch.empty(
        [lhs.shape[0], lhs.shape[1], rhs_unpacked.shape[0]],
        dtype=torch.float16,
    )
    # TODO: fix quantization so the flatten is not necessary
    return wave_mxfp4_bmm(
        lhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        lhs_unpacked.d.squeeze(-1),
        rhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        rhs_unpacked.d.squeeze(-1),
        output,
    )


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank.asm")
def matmul_generic_tensor_block_scaled_fp4_asm(
    lhs, rhs: QuantizedTensor, *, transpose_rhs: bool
):
    """Generic kernel for FP4 E2M1 block scaled layouts using preshuffle ASM kernel.

    This version should only be used if the irpa file was generated with `--apply-shuffle`.
    This kernel expects that the weights have been transformed at IRPA generation time
    to account for the different indexing pattern. However, this kernel offers a modest
    performance gain over a non-shuffled version."""

    """Generic kernel for FP4 E2M1 block scaled layouts."""
    if rhs.layout_type is not BlockScaledFp4Layout:
        return NotImplemented

    if not torch.compiler.is_compiling():
        lhs = unbox_tensor(lhs)
        rhs = unbox_tensor(rhs)
        return matmul(lhs, rhs, transpose_rhs=transpose_rhs)

    # flatten lhs [b, m, k] -> [b * m, k]
    if len(lhs.shape) == 3:
        lhs_flatten = lhs.view(-1, lhs.size(-1))

    lhs_flatten = unbox_tensor(lhs_flatten)
    if not transpose_rhs:
        return NotImplemented
    rhs_unpacked = rhs.unpack()
    quantizer = DynamicFp4BlockQuantizer(
        block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
    )
    lhs_quantized = quantizer.quantize(lhs_flatten)
    lhs_unpacked = lhs_quantized.unpack()

    out = asm_fp4_gemm(
        lhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        rhs_unpacked.qs_bit_packed.flatten(start_dim=-2),
        lhs_unpacked.d.squeeze(-1),
        rhs_unpacked.d.squeeze(-1),
    )
    # [b * m, n] -> [b, m, n]
    return out.view(lhs.shape[0], lhs.shape[1], -1)


@matmul.override(Tensor, QuantizedTensor, impl_name="amdsharktank")
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


@sum.override(AllNotOfType(AnyTensor))
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
