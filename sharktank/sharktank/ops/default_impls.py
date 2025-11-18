# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides of the standard ops for normal torch and
# generic primitive/quantized types.

from typing import Optional, List, Sequence, Union, Tuple

import torch
from torch import Tensor, dtype
import torch.nn.functional as F
from numbers import Number

from amdsharktank.types import (
    DefaultPrimitiveTensor,
    PrimitiveTensor,
    DefaultPrimitiveTensor,
    QuantizedTensor,
    InferenceTensor,
    PlanarQuantizedTensor,
    BlockScaledI4Layout,
    BlockScaledLayout,
    SplitPrimitiveTensor,
    TensorScaledLayout,
    QuantizedLayout,
    unbox_tensor,
    AnyTensor,
)

from amdsharktank.kernels.topk import iree_topk
from amdsharktank.ops.shape import normalize_negative_dim

from ._registry import AllOfType, AllOfExprs, AllOfExprsVariadic, IsOfType, AnyType
from .signatures import *
import iree.turbine.ops.iree


@abs.override(Tensor)
def abs_default(tensor: Tensor) -> Tensor:
    return torch.abs(unbox_tensor(tensor))


@arange.override()
def arange_default(
    *start_end_step,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> DefaultPrimitiveTensor:
    if devices is not None:  # Replicated variant should be used.
        return NotImplemented

    return torch.arange(*start_end_step, dtype=dtype, device=device)


@argmax.override(Tensor)
def argmax_default(
    x: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
) -> None:
    if chunk_size is None:
        return torch.argmax(unbox_tensor(x), dim=dim, keepdim=keepdim)

    return _split_argmax(
        unbox_tensor(x),
        dim=dim,
        keepdim=keepdim,
        chunk_size=chunk_size,
    )


def _split_argmax(input_tensor, dim, keepdim: bool = False, chunk_size: int = 128):
    input_tensor = unbox_tensor(input_tensor)
    dim = dim if dim >= 0 else input_tensor.dim() + dim

    if input_tensor.shape[dim] % chunk_size != 0:
        raise ValueError(
            "dim's size must be a multiple of chunk_size.\n"
            f"Dim Size: {dim}\n"
            f"Chunk Size: {chunk_size}\n"
        )

    n_chunks = input_tensor.shape[dim] // chunk_size
    tensor_split = unflatten(input_tensor, dim, (n_chunks, chunk_size))

    argmax_1 = argmax(tensor_split, dim + 1)
    argmax_expanded = unsqueeze(argmax_1, dim + 1)

    max_vals = gather(tensor_split, dim + 1, argmax_expanded)
    max_vals = squeeze(max_vals, dim + 1)

    argmax_2 = argmax(max_vals, dim)
    index_shape = list(argmax_1.shape)
    index_shape[dim] = 1
    argmax_2_expanded = argmax_2.unsqueeze(dim)

    final_index_in_chunk = gather(argmax_1, dim, argmax_2_expanded)
    final_index = argmax_2_expanded * tensor_split.shape[dim + 1] + final_index_in_chunk

    final_index = squeeze(final_index, dim)

    if keepdim:
        final_index = unsqueeze(final_index, dim)

    return final_index


@cat.override(AllOfType(Tensor, PrimitiveTensor))
def cat_default(tensors: Sequence[Tensor | PrimitiveTensor], dim: int):
    result = torch.cat([unbox_tensor(t) for t in tensors], dim)
    if isinstance(tensors[0], PrimitiveTensor):
        result = DefaultPrimitiveTensor(data=result)
    return result


@chunk.override(Tensor)
def chunk_default(
    tensor: Tensor | PrimitiveTensor, chunks: int, dim: int = 0
) -> tuple[Tensor, ...]:
    return torch.chunk(unbox_tensor(tensor), chunks, dim)


# conv2d


def conv2d_default(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype: Optional[torch.dtype],
):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    if bias is not None:
        bias = unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return F.conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv2d.override(Tensor, Tensor, Tensor, auto_dequant=True)(conv2d_default)
conv2d.override(Tensor, Tensor, auto_dequant=True)(conv2d_default)

# conv3d


def conv3d_default(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype: Optional[torch.dtype],
):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    if bias is not None:
        bias = unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return F.conv3d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv3d.override(Tensor, Tensor, Tensor, auto_dequant=True)(conv3d_default)
conv3d.override(Tensor, Tensor, auto_dequant=True)(conv3d_default)


# conv1d


def conv1d_default(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype: Optional[torch.dtype],
):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    if bias is not None:
        bias = unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return F.conv1d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv1d.override(Tensor, Tensor, Tensor, auto_dequant=True)(conv1d_default)
conv1d.override(Tensor, Tensor, auto_dequant=True)(conv1d_default)


@cos.override(Tensor)
def cos_default(tensor: Tensor) -> Tensor:
    return torch.cos(unbox_tensor(tensor))


# Einsum
def mk_menk_men(inputs, weights):
    # batch dims: m, lhs pdims: none, lhs rdims: k, rhs pdims: en, rhs rdims: k
    inputs = inputs.unsqueeze(1)
    weights_shape = weights.shape
    weights = weights.view(
        weights_shape[0], weights_shape[1] * weights_shape[2], weights_shape[3]
    )
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


def mek_menk_men(inputs, weights):
    # batch dims: me, lhs pdims: none, lhs rdims: k, rhs pdims: n, rhs rdims: k
    inputs_shape = inputs.shape
    inputs = inputs.view(inputs_shape[0] * inputs_shape[1], 1, inputs_shape[2])
    weights_shape = weights.shape
    weights = weights.view(
        weights_shape[0] * weights_shape[1], weights_shape[2], weights_shape[3]
    )
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


def me_men_men(inputs, weights):
    # batch dims: me, lhs pdims: none, lhs rdims: none, rhs pdims: n, rhs rdims: none
    inputs_shape = inputs.shape
    inputs = inputs.view(inputs_shape[0] * inputs_shape[1], 1, 1)
    weights_shape = weights.shape
    weights = weights.view(weights_shape[0] * weights_shape[1], weights_shape[2], 1)
    result = matmul(inputs, weights, transpose_rhs=True)
    result = result.view(weights_shape[0], weights_shape[1], weights_shape[2])
    return result


def bqhmd_bkhmd_bhmqk(q, k):
    # q: (B,q,H,M,D), k: (B,k,H,M,D) -> (B,H,M,q,k)
    batch_seq, q_len, n_heads, q_mul, d_head = q.shape
    batch_seq, kv_len, n_heads, q_mul, d_head = k.shape
    q_mat = q.permute(0, 2, 3, 1, 4)  # (B, H, M, q_len,  D)
    k_mat = k.permute(0, 2, 3, 1, 4)  # (B, H, M, kv_len, D)
    q_mat = q_mat.reshape(batch_seq * n_heads * q_mul, q_len, d_head)
    k_mat = k_mat.reshape(batch_seq * n_heads * q_mul, kv_len, d_head)
    qk = matmul(q_mat, k_mat.transpose(1, 2))  # (BHM, q_len, kv_len)
    qk = qk.view(batch_seq, n_heads, q_mul, q_len, kv_len)  # (B, H, M, q, k)
    return qk


def bhmqk_bkhmd_bqhmd(w, v):
    # w: (B,H,M,q,k), v: (B,k,H,M,D) -> (B,H,M,q,D)
    batch_seq, n_heads, q_mul, q_len, kv_len = w.shape
    batch_seq, kv_len, n_heads, q_mul, d_head = v.shape
    w_mat = w.reshape(batch_seq * n_heads * q_mul, q_len, kv_len)
    v_mat = v.permute(0, 2, 3, 1, 4)
    v_mat = v_mat.reshape(batch_seq * n_heads * q_mul, kv_len, d_head)
    wv = matmul(w_mat, v_mat)
    wv = wv.view(batch_seq, n_heads, q_mul, q_len, d_head).permute(0, 3, 1, 2, 4)
    return wv


@einsum_2args.override(AllOfType(Tensor, PrimitiveTensor, QuantizedTensor))
def einsum_2args(input0, input1, einsum_str):
    # Special optimized einsum kernels that lower to batch matmul
    if einsum_str == "mk,menk->men":
        return mk_menk_men(input0, input1)
    elif einsum_str == "mek,menk->men":
        return mek_menk_men(input0, input1)
    elif einsum_str == "me,men->men":
        return me_men_men(input0, input1)
    elif einsum_str == "bqhmd,bkhmd->bhmqk":
        return bqhmd_bkhmd_bhmqk(input0, input1)
    elif einsum_str == "bhmqk,bkhmd->bqhmd":
        return bhmqk_bkhmd_bqhmd(input0, input1)
    # Default non-QuantizedTensor einsum
    if not isinstance(input1, QuantizedTensor):
        return torch.einsum(einsum_str, unbox_tensor(input0), unbox_tensor(input1))
    # Fallback to other kernels
    return NotImplemented


# Elementwise
@elementwise.override(Tensor)
def elementwise_unary(operator, x, *args, **kwargs):
    x = unbox_tensor(x)
    return operator(x, *args, **kwargs)


@elementwise.override(
    AllOfExprs(
        IsOfType(Tensor, PrimitiveTensor), IsOfType(Tensor, PrimitiveTensor, Number)
    )
)
def elementwise_binary(operator, x, y, *args, **kwargs):
    x = unbox_tensor(x)
    if isinstance(y, PrimitiveTensor):
        y = unbox_tensor(y)
    return operator(x, y, *args, **kwargs)


@elementwise.override(
    AllOfExprs(
        IsOfType(Tensor, PrimitiveTensor),
        IsOfType(Tensor, PrimitiveTensor, Number),
        IsOfType(Tensor, PrimitiveTensor, Number),
    )
)
def elementwise_ternary(operator, x, y, z, *args, **kwargs):
    x = unbox_tensor(x)
    if isinstance(y, PrimitiveTensor):
        y = unbox_tensor(y)
    if isinstance(z, PrimitiveTensor):
        z = unbox_tensor(z)
    return operator(x, y, z, *args, **kwargs)


# Embedding Lookup
@embedding_lookup.override(Tensor, Tensor)
def embedding_lookup_default(input, embedding_matrix, dtype: Optional[dtype]):
    return F.embedding(unbox_tensor(input), unbox_tensor(embedding_matrix).to(dtype))


@embedding_lookup.override(Tensor, QuantizedTensor)
def embedding_lookup_Tensor_QuantizedTensor(
    input, embedding_matrix: QuantizedTensor, dtype: Optional[dtype]
):
    dequant = embedding_matrix.unpack().dequant(dtype=dtype)
    return F.embedding(unbox_tensor(input), dequant)


@equal.override(AllOfType(Tensor, InferenceTensor))
def equal_default(a: Tensor | InferenceTensor, b: Tensor | InferenceTensor) -> bool:
    return torch.equal(unbox_tensor(a), unbox_tensor(b))


@expand.override(Tensor)
def expand_default(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    return unbox_tensor(tensor).expand(*shape)


@expand.override(PlanarQuantizedTensor)
def expand_quantized_dispatcher(
    tensor: PlanarQuantizedTensor, shape: List[int]
) -> PlanarQuantizedTensor:
    return expand(tensor.unpack(), shape)


@expand.override(TensorScaledLayout)
def expand_tensor_scaled_layout(
    layout: TensorScaledLayout, shape: List[int]
) -> PlanarQuantizedTensor:
    new_qs = layout._qs.expand(*shape)
    new_layout = TensorScaledLayout(
        shape=new_qs.shape,
        d=layout._d,
        qs=new_qs,
        m=layout._m,
        dtype=layout.dtype,
    )
    return PlanarQuantizedTensor(shape=new_qs.shape, layout=new_layout)


@flatten.override(Tensor)
def flatten_default(
    input: Union[Tensor, PrimitiveTensor], start_dim: int, end_dim: int
) -> Tensor:
    return torch.flatten(unbox_tensor(input), start_dim, end_dim)


@flatten.override(PlanarQuantizedTensor)
def flatten_quantized_dispatcher(
    tensor: PlanarQuantizedTensor, start_dim: int, end_dim: int
) -> PlanarQuantizedTensor:
    return flatten(tensor.unpack(), start_dim, end_dim)


@flatten.override(TensorScaledLayout)
def flatten_tensor_scaled_layout(
    layout: TensorScaledLayout, start_dim: int, end_dim: int
) -> PlanarQuantizedTensor:
    new_qs = torch.flatten(layout._qs, start_dim, end_dim)
    new_layout = TensorScaledLayout(
        shape=new_qs.shape,
        d=layout._d,
        qs=new_qs,
        m=layout._m,
        dtype=layout.dtype,
    )
    return PlanarQuantizedTensor(shape=new_qs.shape, layout=new_layout)


@full.override()
def full_default(
    size: Sequence[int],
    fill_value: Number,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> DefaultPrimitiveTensor:
    if devices is not None:  # Replicated variant should be used.
        return NotImplemented

    return torch.full(size, fill_value, dtype=dtype, device=device)


@gather.override(Tensor, Tensor)
def gather_default(
    input: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
) -> Tensor:
    return torch.gather(unbox_tensor(input), dim, unbox_tensor(index))


@extract_slice.override(AllOfType(Tensor, PrimitiveTensor))
def extract_slice_default(tensor, key):
    return unbox_tensor(tensor)[key]


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
        if unpacked._m.dim() == 0:
            m = unpacked._m
        else:
            m = unpacked._m[key]
        shape = qs.shape
        layout = TensorScaledLayout(shape=shape, d=d, qs=qs, m=m)
        return PlanarQuantizedTensor(shape=shape, layout=layout)
    return NotImplemented


@gemm.override(AllOfType(Tensor, InferenceTensor))
def gemm(
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor],
    alpha: Optional[Union[Number, AnyTensor]],
    beta: Optional[Union[Number, AnyTensor]],
    transa: bool,
    transb: bool,
) -> bool:
    if transa:
        a = a.T
    if transb:
        b = b.T
    res = matmul(a, b)
    if alpha is not None:
        res = alpha * res
    if c is not None:
        if beta is not None:
            res = res + beta * c
        else:
            res = res + c
    return res


# Group norm.
@group_norm_affine.override(Tensor, Tensor, Tensor)
def group_norm_affine_default(input, weight, bias, *, num_groups, eps):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    return F.group_norm(input, num_groups=num_groups, weight=weight, bias=bias, eps=eps)


@index_copy_.override(Tensor, Tensor, Tensor)
def index_copy__default(
    inout: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
    tensor: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    index = unbox_tensor(index)
    tensor = unbox_tensor(tensor)
    inout_as_torch = unbox_tensor(inout)
    if not torch.compiler.is_compiling() and inout_as_torch.dtype in [
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,
    ]:
        # PyTorch/PyTorch ROCm does not have eager implementation for various dtypes
        # for CPU/GPU.
        # We need to view as int8 before performing the operation.
        # We still want to avoid the bitcasts during export as the IREE compiler has
        # trouble fusing them.
        # We could maybe be more picky in selecting this path depending on the exact
        # GPU arch and PyTorch ROCm version to determine if there is support to call
        # directly.
        inout_as_torch = inout_as_torch.view(dtype=torch.int8)
        tensor = tensor.view(dtype=torch.int8)
    inout_as_torch.index_copy_(dim, index, tensor)
    return inout


@index_put_.override(AllOfType(Tensor, PrimitiveTensor))
def index_put__default(
    inout: Union[Tensor, PrimitiveTensor],
    indices: Tuple[Union[Tensor, PrimitiveTensor]],
    values: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    indices = tuple(unbox_tensor(index) for index in indices)
    inout_as_torch = unbox_tensor(inout)
    values = unbox_tensor(values)
    if (
        not torch.compiler.is_compiling()
        and inout_as_torch.is_cpu
        and inout_as_torch.dtype in [torch.float8_e4m3fnuz, torch.float8_e4m3fn]
    ):
        # PyTorch does not have eager implementation for float8_e4m3fnuz in CPU.
        # We need to view as int8 before performing the operation.
        # We still want to avoid the bitcasts during export as the IREE compiler has
        # trouble fusing them.
        inout_as_torch = inout_as_torch.view(dtype=torch.int8)
        values = values.view(dtype=torch.int8)

    inout_as_torch.index_put_(indices, values)
    return inout


@index_select.override(Tensor, Tensor)
def index_select_default(
    tensor: Union[Tensor, PrimitiveTensor],
    dim: int,
    index: Union[Tensor, PrimitiveTensor],
) -> Union[Tensor, PrimitiveTensor]:
    return torch.index_select(unbox_tensor(tensor), dim, unbox_tensor(index))


@interpolate.override(Tensor)
def interpolate_default(
    input: Tensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> Tensor:
    return torch.nn.functional.interpolate(
        input=unbox_tensor(input),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def layer_norm_default(input, weight, bias, *, eps, normalized_shape):
    input = unbox_tensor(input)
    if weight is not None:
        weight = unbox_tensor(weight)
    if bias is not None:
        bias = unbox_tensor(bias)
    if normalized_shape is None:
        assert weight is not None
        normalized_shape = weight.shape
    return F.layer_norm(
        input, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


layer_norm.override(Tensor)(layer_norm_default)
layer_norm.override(Tensor, Tensor)(layer_norm_default)
layer_norm.override(Tensor, Tensor, Tensor)(layer_norm_default)


# Linear
def linear_default(input, weight, bias, *, accum_dtype, matmul_impl) -> Tensor:
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = None if bias is None else unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(dtype=input.dtype)
    result = matmul(input, weight, transpose_rhs=True, impl=matmul_impl)
    if bias is not None:
        result = result + bias
    return result


linear.override(Tensor, Tensor, auto_dequant=True)(linear_default)
linear.override(Tensor, Tensor, Tensor, auto_dequant=True)(linear_default)


@log.override(Tensor)
def log_default(tensor: Tensor) -> Tensor:
    return torch.log(unbox_tensor(tensor))


@logical_or.override(AllOfType(Tensor, PrimitiveTensor))
def logical_or_default(
    input: Tensor | PrimitiveTensor, other: Tensor | PrimitiveTensor
) -> Tensor | PrimitiveTensor:
    return torch.logical_or(unbox_tensor(input), unbox_tensor(other))


@masked_fill.override(AllOfType(Tensor, PrimitiveTensor))
def masked_fill_default(
    tensor: Tensor | PrimitiveTensor,
    mask: Tensor | PrimitiveTensor,
    value: Number,
) -> Union[Tensor, PrimitiveTensor]:
    tensor = unbox_tensor(tensor)
    mask = unbox_tensor(mask)
    return tensor.masked_fill(mask, value)


# Matmul
@matmul.override(Tensor, Tensor, auto_dequant=True, impl_name="torch")
def matmul_default(lhs, rhs, *, transpose_rhs: bool) -> Tensor:
    lhs = unbox_tensor(lhs)
    rhs = unbox_tensor(rhs)
    if transpose_rhs:
        rhs = rhs.mT
    rhs = rhs.to(lhs.dtype)

    if len(lhs.shape) > 2 and len(rhs.shape) < 3:
        bdims = lhs.shape[:-1]
        lhs = torch.flatten(lhs, 0, -2)
        mm = torch.matmul(lhs, rhs)
        return torch.unflatten(mm, 0, bdims)

    return torch.matmul(lhs, rhs)


@mean.override(Tensor)
def mean_default(
    x: Tensor, dim: Union[int, List[int]], keepdim: bool, *, dtype: torch.dtype
) -> None:
    return torch.mean(unbox_tensor(x), dim=dim, keepdim=keepdim, dtype=dtype)


@module_register_buffer.override(torch.nn.Module, Tensor)
def module_register_buffer_default(
    module: torch.nn.Module, name: str, tensor: Union[Tensor, InferenceTensor]
) -> None:
    return module.register_buffer(name, unbox_tensor(tensor))


@ones.override()
def ones_default(
    *size,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> DefaultPrimitiveTensor:
    if devices is not None:  # Replicated variant should be used.
        return NotImplemented

    return torch.ones(*size, device=device, dtype=dtype)


@ones_like.override(AllOfType(Tensor, PrimitiveTensor))
def ones_like_default(
    tensor: Union[Tensor, PrimitiveTensor],
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> Tensor:
    return torch.ones_like(unbox_tensor(tensor), dtype=dtype, device=device)


@repeat.override(Tensor)
def repeat_default(input: Union[Tensor, PrimitiveTensor], *sizes: List[int]) -> Tensor:
    return unbox_tensor(input).repeat(*sizes)


@reshape.override(Tensor)
def reshape_default(input: Union[PrimitiveTensor, Tensor], shape: List[int]) -> Tensor:
    return torch.reshape(unbox_tensor(input), shape)


# RMS norm
@rms_norm.override(AllOfType(Tensor, InferenceTensor))
def rms_norm_default(
    x, weight, *, epsilon: float, orig_dtype: Union[None, torch.dtype]
) -> Tensor:
    if orig_dtype is None:
        orig_dtype = x.dtype
    variance = x.pow(2).mean(-1, keepdim=True)
    output = x * elementwise(torch.rsqrt, variance + epsilon)
    output = elementwise(torch.mul, weight, to(output, orig_dtype))
    return output


@rms_norm.override(Tensor, QuantizedTensor)
def rms_norm_Tensor_QuantizedTensor(
    x, weight: PrimitiveTensor, *, epsilon: float, orig_dtype: Union[None, torch.dtype]
) -> Tensor:
    x = unbox_tensor(x)
    weight = weight.unpack().dequant(x.dtype)
    return rms_norm_default(x, weight, epsilon=epsilon, orig_dtype=orig_dtype)


@pad.override(Tensor)
def pad_default(
    input: Union[Tensor, PrimitiveTensor],
    _pad: Sequence[int],
    mode: str = None,
    value: Optional[float] = None,
) -> Tensor:
    return F.pad(unbox_tensor(input), _pad, mode=mode, value=value)


@permute.override(Tensor)
def permute(tensor: Tensor, dims: List[int]):
    torch_tensor = unbox_tensor(tensor)
    return torch.permute(torch_tensor, dims)


@scatter_.override(
    AllOfExprs(
        IsOfType(Tensor, PrimitiveTensor),
        IsOfType(Tensor, PrimitiveTensor),
        IsOfType(Tensor, PrimitiveTensor, Number),
    )
)
def scatter__default(
    inout: Tensor | PrimitiveTensor,
    dim: int,
    index: Tensor | PrimitiveTensor,
    src: Tensor | PrimitiveTensor | Number,
    *,
    reduce: str | None = None,
) -> Tensor:
    inout = unbox_tensor(inout)
    index = unbox_tensor(index)
    if isinstance(src, (torch.Tensor, PrimitiveTensor)):
        src = unbox_tensor(src)
    if reduce is not None:
        inout.scatter_(dim, index, src, reduce=reduce)
    else:
        inout.scatter_(dim, index, src)
    return inout


@scatter_add.override(AllOfType(Tensor, PrimitiveTensor))
def scatter_add_default(
    input: Tensor | PrimitiveTensor,
    dim: int,
    index: Tensor | PrimitiveTensor,
    src: Tensor | PrimitiveTensor,
) -> Tensor:
    input = unbox_tensor(input)
    index = unbox_tensor(index)
    src = unbox_tensor(src)
    return torch.scatter_add(input, dim, index, src)


@sigmoid.override(Tensor)
def sigmoid_default(tensor: Tensor) -> Tensor:
    return tensor.sigmoid()


@sin.override(Tensor)
def sin_default(tensor: Tensor) -> Tensor:
    return torch.sin(unbox_tensor(tensor))


@softmax.override(Tensor)
def softmax_default(
    tensor: Union[Tensor, PrimitiveTensor],
    dim: Optional[int],
    dtype: Optional[torch.dtype],
) -> Tensor:
    return F.softmax(unbox_tensor(tensor), dim=dim, dtype=dtype)


@split.override(Tensor)
def split_default(
    tensor: Tensor | PrimitiveTensor,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[Tensor, ...]:
    return torch.split(unbox_tensor(tensor), split_size_or_sections, dim)


@split.override(IsOfType(QuantizedTensor, SplitPrimitiveTensor))
def split_via_extract_slice(
    tensor: QuantizedTensor | SplitPrimitiveTensor,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[QuantizedTensor | SplitPrimitiveTensor, ...]:
    dim = normalize_negative_dim(tensor, dim)
    dim_size = tensor.shape[dim]
    if isinstance(split_size_or_sections, int):
        sections = [split_size_or_sections] * (dim_size // split_size_or_sections)
        reminder = dim_size % split_size_or_sections
        if reminder != 0:
            sections.append(reminder)
        return split(tensor, sections, dim)

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
        res.append(extract_slice(tensor, slice_))
    return tuple(res)


@swiglu.override(Tensor)
def swiglu_default(
    x: Tensor, *, alpha: float = 1.702, limit: float | None = None
) -> Tensor:
    x = unbox_tensor(x)
    if x.size(-1) % 2 != 0:
        raise ValueError(f"SwiGLU expects even last dim, got {x.size(-1)}")

    # Split interleaved channels using NumPy-style slicing (start:stop:step).
    x_glu = x[..., ::2]  # even indices along the last dimension
    x_lin = x[..., 1::2]  # odd indices along the last dimension

    if limit is not None:
        x_glu = x_glu.clamp(min=None, max=limit)
        x_lin = x_lin.clamp(min=-limit, max=limit)
    # SwiGLU: swish(alpha * a) * (b + 1)
    alpha = tensor(alpha, dtype=x.dtype)
    out_glu = x_glu * sigmoid(alpha * x_glu)
    return out_glu * (x_lin + 1)


@to.override(Tensor)
def to_default(tensor: Tensor, *args, **kwargs) -> PrimitiveTensor:
    return DefaultPrimitiveTensor(data=unbox_tensor(tensor).to(*args, **kwargs))


@trace_tensor.override(AllOfExprsVariadic(IsOfType(Tensor, InferenceTensor)))
def trace_tensor(key: str, *tensors: tuple[AnyTensor, ...]):
    if len(tensors) != 1:
        raise ValueError("Tracing more than one tensor at a time is not supported.")
    tensor = unbox_tensor(unshard(tensors[0]))
    iree.turbine.ops.iree.trace_tensor(key, tensor)


@transfer_to_logical_device.override(Tensor)
def transfer_to_logical_device_default(tensor: Tensor, ordinal: int):
    transfered = iree.turbine.ops.iree.transfer_to_logical_device(
        f"{ordinal}", unbox_tensor(tensor)
    )
    if isinstance(tensor, DefaultPrimitiveTensor):
        transfered = DefaultPrimitiveTensor(data=transfered, name=tensor.name)
    return transfered


@barrier_on_logical_device.override(Tensor)
def barrier_on_device_default(tensor: Tensor, ordinal: int):
    barriered = iree.turbine.ops.iree.barrier_on_logical_device(
        f"{ordinal}", unbox_tensor(tensor)
    )
    if isinstance(tensor, DefaultPrimitiveTensor):
        barriered = DefaultPrimitiveTensor(data=barriered, name=tensor.name)
    return barriered


@transpose.override(Tensor)
def transpose_default(
    tensor: Union[Tensor, PrimitiveTensor], dim0: int, dim1: int
) -> Union[Tensor, PrimitiveTensor]:
    transposed = torch.transpose(unbox_tensor(tensor), dim0, dim1)
    if isinstance(tensor, PrimitiveTensor):
        transposed = DefaultPrimitiveTensor(data=transposed, name=tensor.name)
    return transposed


@transpose.override(PlanarQuantizedTensor)
def transpose_PlanarQuantizedTensor(
    tensor: PlanarQuantizedTensor, dim0: int, dim1: int
) -> PlanarQuantizedTensor:
    layout = tensor.unpack()

    if isinstance(layout, BlockScaledLayout):
        last_index = [-1, len(layout.shape) - 1]
        if dim0 in last_index or dim1 in last_index:
            raise ValueError("Cannot transpose last dim of BlockScaledLayout tensors.")

    new_planes = {}
    for name, plane in layout.planes.items():
        if len(plane.shape) < 2:
            new_planes[name] = plane
        else:
            new_planes[name] = plane.transpose(dim0, dim1)

    new_shape = list(layout.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

    new_layout = layout.__class__.create(
        shape=new_shape,
        metadata=layout.metadata,
        planes=new_planes,
    )
    return PlanarQuantizedTensor(shape=new_layout.shape, layout=new_layout)


# Sharded default impls (do nothing).


@reduce_scatter.override(Tensor)
def reduce_scatter_unsharded(
    tensor: AnyTensor, scatter_dim: int
) -> Tensor | InferenceTensor:
    return tensor


@sharded_cat.override(Tensor)
def sharded_cat_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@sharded_gather.override(Tensor)
def sharded_gather_unsharded(input):
    return [input]


@sharded_sum.override(Tensor)
def sharded_sum_unsharded(tensor: Tensor, root_rank: int) -> Tensor:
    if root_rank != 0:
        raise ValueError(
            f"sharded_sum destination rank {root_rank} is invalid for"
            f" tensor of type {type(tensor)}. Only rank of 0 is allowed"
        )
    return unbox_tensor(tensor)


@sum.override(AllOfType(Tensor, PrimitiveTensor))
def sum_default(
    input: Tensor | PrimitiveTensor,
    dim: Union[int, List[int]] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype,
) -> Tensor:
    return torch.sum(unbox_tensor(input), dim=dim, keepdim=keepdim, dtype=dtype)


@unflatten.override(Tensor)
def unflatten_default(
    input: Union[Tensor, PrimitiveTensor], dim: int, sizes: Tuple[int]
) -> Tensor:
    return torch.unflatten(unbox_tensor(input), dim, sizes)


@unsqueeze.override(Tensor)
def unsqueeze_default(tensor: Union[Tensor, PrimitiveTensor], dim: int) -> Tensor:
    return torch.unsqueeze(unbox_tensor(tensor), dim)


@unsqueeze.override(PlanarQuantizedTensor)
def unsqueeze_quantized_dispatcher(
    tensor: PlanarQuantizedTensor, dim: int
) -> PlanarQuantizedTensor:
    return unsqueeze(tensor.unpack(), dim)


@unsqueeze.override(TensorScaledLayout)
def unsqueeze_tensor_scaled_layout(
    layout: TensorScaledLayout, dim: int
) -> PlanarQuantizedTensor:
    new_qs = layout._qs.unsqueeze(dim)
    new_layout = TensorScaledLayout(
        shape=new_qs.shape,
        d=layout._d,
        qs=new_qs,
        m=layout._m,
        dtype=layout.dtype,
    )
    return PlanarQuantizedTensor(shape=new_qs.shape, layout=new_layout)


@squeeze.override(AllOfType(AnyTensor, PrimitiveTensor))
def squeeze_default(tensor, dim: Optional[int] = None) -> AnyTensor:
    if dim is None:
        return torch.squeeze(unbox_tensor(tensor))
    else:
        return torch.squeeze(unbox_tensor(tensor), dim)


def tensor_default(
    data: AnyTensor | Number,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    if devices is not None:
        return NotImplemented

    if isinstance(data, AnyTensor):
        data = unbox_tensor(data)
    return torch.tensor(data, dtype=dtype, device=device)


tensor.override()(tensor_default)
tensor.override(Tensor)(tensor_default)


@topk.override(AllOfType(Tensor, PrimitiveTensor))
def topk_default(
    tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    chunk_size: Optional[int] = None,
    use_linalgext_topk: bool = False,
) -> tuple[Tensor, Tensor]:

    if use_linalgext_topk:
        assert largest
        assert not sorted
        assert dim == len(tensor.shape) - 1 or dim == -1
        bs_shape = tensor.shape[:-1]

        tensor = unbox_tensor(tensor.flatten(0, -2))
        flat_bs = tensor.shape[0]

        indices = torch.arange(tensor.shape[1], dtype=torch.int32)[None, :].repeat(
            tensor.shape[0], 1
        )

        if chunk_size:
            tensor = tensor.unflatten(dim, (chunk_size, tensor.shape[-1] // chunk_size))
            tensor = tensor.flatten(0, 1)
            indices = indices.unflatten(
                dim, (chunk_size, indices.shape[-1] // chunk_size)
            )
            indices = indices.flatten(0, 1)

        values, indices = iree_topk(tensor, indices, k=k)

        if chunk_size:
            values = values.unflatten(0, (flat_bs, chunk_size)).flatten(1)
            indices = indices.unflatten(0, (flat_bs, chunk_size)).flatten(1)
            values, indices = iree_topk(values, indices, k=k)

        values = unflatten(values, 0, bs_shape)
        indices = unflatten(indices, 0, bs_shape)
        return values, indices.to(torch.int64)

    if chunk_size is not None:
        return _split_topk(
            tensor, k, dim, largest, sorted, chunk_size, use_linalgext_topk
        )

    result = torch.topk(
        unbox_tensor(tensor), k=k, dim=dim, largest=largest, sorted=sorted
    )
    return result.values, result.indices


def _split_topk(
    tensor: Tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    chunk_size: int,
    use_linalgext_topk: bool,
) -> Tuple[Tensor, Tensor]:
    """Find the `topk` of a tensor using `split_k` strategy for better perf.

    Args:
        tensor (Tensor): Tensor to take `topk` of.
        k (int): Number of max tokens to select.
        dim (int): Dim to take along.
        largest (bool): Return largest or smallest indices.
        sorted (bool): Return results in sorted order or not.
        chunk_size (int): Size to split groups into.

    Raises:
        ValueError: k must be positive
        ValueError: dim length must be a multiple of chunk_size

    Returns:
        Tuple[Tensor, Tensor]: Selected values and indices.
    """
    # TODO(stbaione): Explore more algorithms, like `grouped_argmax` for better perf.
    tensor = unbox_tensor(tensor)

    if k <= 0:
        raise ValueError("k must be positive")
    dim = dim if dim >= 0 else tensor.dim() + dim

    if tensor.shape[dim] % chunk_size:
        raise ValueError("dim length must be a multiple of chunk_size")

    n_chunks = tensor.shape[dim] // chunk_size
    tensor_unflattened = unflatten(tensor, dim, (n_chunks, chunk_size))

    vals_local, idx_local = topk(
        tensor_unflattened,
        k,
        dim=dim + 1,
        largest=largest,
        sorted=sorted,
        use_linalgext_topk=use_linalgext_topk,
    )

    vals_flat = flatten(vals_local, start_dim=dim, end_dim=dim + 1)
    idx_flat = flatten(idx_local, start_dim=dim, end_dim=dim + 1)

    vals_out, flat_idx = topk(
        vals_flat,
        k,
        dim=dim,
        largest=largest,
        sorted=sorted,
        use_linalgext_topk=use_linalgext_topk,
    )

    chunk_idx = flat_idx // k

    local_pos = gather(idx_flat, dim, flat_idx)
    idx_out = local_pos + chunk_idx * chunk_size

    return vals_out, idx_out


@view.override(Tensor)
def view_default(
    tensor: Union[Tensor, PrimitiveTensor],
    shape: List[int] | None,
    dtype: torch.dtype | None,
) -> Tensor:
    assert (shape is None) ^ (
        dtype is None
    ), "Exactly one of shape or dtype must be provided"
    if shape is not None:
        return unbox_tensor(tensor).view(*shape)
    else:
        return unbox_tensor(tensor).view(dtype)


@view.override(PlanarQuantizedTensor)
def view_quantized_dispatcher(tensor: PlanarQuantizedTensor, shape, dtype):
    return view(tensor.unpack(), shape, dtype)


@view.override(TensorScaledLayout)
def view_tensor_scaled_layout(layout: TensorScaledLayout, shape, dtype):
    if dtype:
        return NotImplemented
    new_qs = layout._qs.view(shape)
    new_layout = TensorScaledLayout(
        shape=shape, d=layout._d, qs=new_qs, m=layout._m, dtype=dtype
    )
    return PlanarQuantizedTensor(shape=shape, layout=new_layout)


@view.override(BlockScaledI4Layout)
def view_block_scaled_i4_layout(layout: BlockScaledI4Layout, shape, dtype):
    if dtype:
        return NotImplemented

    shape = list(shape)
    new_d = layout._d.view(shape[:-1] + [shape[-1] // 32, 1])
    qs_shape = shape[:-1] + [shape[-1] // 32, 16]
    new_qs = layout._qs.view(qs_shape)
    if layout.m is not None:
        new_m = layout.m.view(shape[:-1] + [shape[-1] // 32, 1])
    else:
        new_m = None
    new_layout = BlockScaledI4Layout(shape=shape, d=new_d, qs=new_qs, m=new_m)
    return PlanarQuantizedTensor(shape=shape, layout=new_layout)


@view_as_complex.override(Tensor)
def view_as_complex_default(tensor: Union[Tensor, PrimitiveTensor]) -> Tensor:
    return torch.view_as_complex(unbox_tensor(tensor))


@view_as_real.override(Tensor)
def view_as_real_default(tensor: Union[Tensor, PrimitiveTensor]) -> Tensor:
    return torch.view_as_real(unbox_tensor(tensor))


def where_default(
    condition: Tensor,
    input: Tensor | Number | None = None,
    other: Tensor | Number | None = None,
) -> Tensor | Tuple[Tensor, ...]:
    assert (input is None) == (other is None)

    condition = unbox_tensor(condition)
    input = unbox_tensor(input) if isinstance(input, AnyTensor) else input
    other = unbox_tensor(other) if isinstance(other, AnyTensor) else other
    if input is None:
        return torch.where(condition)
    else:
        return torch.where(condition, input, other)


where.override(Tensor)(where_default)
where.override(Tensor, Tensor)(where_default)
where.override(Tensor, Tensor, Tensor)(where_default)


@zeros.override()
def zeros_default(
    *size,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> DefaultPrimitiveTensor:
    if devices is not None:  # Replicated variant should be used.
        return NotImplemented

    return torch.zeros(*size, dtype=dtype, device=device)


@zeros_like.override(AllOfType(Tensor, PrimitiveTensor))
def zeros_like_default(
    tensor: Union[Tensor, PrimitiveTensor],
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> Tensor:
    return torch.zeros_like(unbox_tensor(tensor), dtype=dtype, device=device)
