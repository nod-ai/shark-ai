# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides of the standard ops for normal torch and
# generic primitive/quantized types.

from typing import Optional, List, Sequence, Union

import torch
from torch import Tensor, dtype
import torch.nn.functional as F
from numbers import Number

from ..types import PrimitiveTensor, QuantizedTensor, InferenceTensor
from ..types.tensors import unbox_tensor, AnyTensor
from ._registry import AllOfType, AllOfExprs, AllOfExprsVariadic, IsOfType
from .signatures import *
import shark_turbine.ops.iree


@cat.override(AllOfType(Tensor, PrimitiveTensor))
def cat_default(tensors: Sequence[Tensor | PrimitiveTensor], dim: int):
    return torch.cat([unbox_tensor(t) for t in tensors], dim)


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

# Elementwise
@elementwise.override(Tensor)
def elementwise_unary(operator, x):
    x = unbox_tensor(x)
    return operator(x)


@elementwise.override(
    AllOfExprs(
        IsOfType(Tensor, PrimitiveTensor), IsOfType(Tensor, PrimitiveTensor, Number)
    )
)
def elementwise_binary(operator, x, y):
    x = unbox_tensor(x)
    if isinstance(y, PrimitiveTensor):
        y = unbox_tensor(y)
    return operator(x, y)


@elementwise.override(
    AllOfExprsVariadic(
        IsOfType(Tensor, InferenceTensor),
        IsOfType(Tensor, InferenceTensor, Number),
        IsOfType(Tensor, InferenceTensor, Number),
    )
)
def elementwise_variadic(operator, x, y, *args):
    """Folds by successively applying the binary operator from left to right until
    exhaustion.

    Match a variable number of tensor/number arguments with at least 3 such arguments.

    Example matches
    ```
    (Tensor, Tensor, Tensor)
    (Tensor, DefaultPrimitiveTensor, float),
    (SplitPrimitiveTensor, ReplicatedTensor, int, Tensor)
    ```

    Will not match
    ```
    (Tensor)
    (Tensor, Tensor)
    (int, Tensor, Tensor)
    ```
    """
    res = elementwise(operator, x, y)
    for arg in args:
        res = elementwise(operator, res, arg)
    return res


# Embedding Lookup
@embedding_lookup.override(Tensor, Tensor)
def embedding_lookup_default(input, embedding_matrix, dtype: dtype):
    return F.embedding(unbox_tensor(input), unbox_tensor(embedding_matrix).to(dtype))


@embedding_lookup.override(Tensor, QuantizedTensor)
def embedding_lookup_Tensor_QuantizedTensor(
    input, embedding_matrix: QuantizedTensor, dtype: dtype
):
    dequant = embedding_matrix.unpack().dequant(dtype=dtype)
    return F.embedding(unbox_tensor(input), dequant)


@equal.override(Tensor, Tensor)
def equal_default(a, b) -> bool:
    return torch.equal(unbox_tensor(a), unbox_tensor(b))


@flatten.override(Tensor)
def flatten_default(
    input: Union[PrimitiveTensor, Tensor], start_dim: int, end_dim: int
) -> Tensor:
    return torch.flatten(unbox_tensor(input), start_dim, end_dim)


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


@layer_norm.override(Tensor, Tensor, Tensor)
def layer_norm_default(input, weight, bias, *, eps):
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = unbox_tensor(bias)
    return F.layer_norm(
        input, normalized_shape=weight.shape, weight=weight, bias=bias, eps=eps
    )


# Linear
def linear_default(input, weight, bias, *, accum_dtype) -> Tensor:
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = None if bias is None else unbox_tensor(bias)
    if weight.dtype != input.dtype:
        weight = weight.to(dtype=input.dtype)
    result = matmul(input, weight, transpose_rhs=True)
    if bias is not None:
        result = result + bias
    return result


linear.override(Tensor, Tensor, auto_dequant=True)(linear_default)
linear.override(Tensor, Tensor, Tensor, auto_dequant=True)(linear_default)


# Matmul
@matmul.override(Tensor, Tensor, auto_dequant=True)
def matmul_default(lhs, rhs, *, transpose_rhs: bool) -> Tensor:
    lhs = unbox_tensor(lhs)
    rhs = unbox_tensor(rhs)
    if transpose_rhs:
        rhs = rhs.T
    return torch.matmul(lhs, rhs.to(lhs.dtype))


# Scaled dot product attention
@scaled_dot_product_attention.override(
    Tensor, Tensor, Tensor, Optional[Tensor], auto_dequant=True
)
def scaled_dot_product_attention(q, k, v, a) -> Tensor:
    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    if a is not None:
        a = unbox_tensor(a)

    # TODO: plumb dropout and is_causal through ops
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=a, dropout_p=0.0, is_causal=False
    )


@reshape.override(Tensor)
def reshape_default(input: Union[PrimitiveTensor, Tensor], shape: List[int]) -> Tensor:
    return torch.reshape(unbox_tensor(input), shape)


# RMS norm
@rms_norm.override(Tensor, Tensor)
def rms_norm_default(x, weight, *, epsilon: float) -> Tensor:
    x = unbox_tensor(x)
    weight = unbox_tensor(weight)
    variance = x.pow(2).mean(-1, keepdim=True)
    output = x * torch.rsqrt(variance + epsilon)
    # The cast here is to match the hf implementation, affects numerics
    output = weight * output.to(weight.dtype)
    return output


@rms_norm.override(Tensor, QuantizedTensor)
def rms_norm_Tensor_QuantizedTensor(
    x, weight: PrimitiveTensor, *, epsilon: float
) -> Tensor:
    x = unbox_tensor(x)
    weight = weight.unpack().dequant(x.dtype)
    return rms_norm_default(x, weight, epsilon=epsilon)


@permute.override(Tensor)
def permute(tensor: Tensor, dims: List[int]):
    torch_tensor = unbox_tensor(tensor)
    return torch.permute(torch_tensor, dims)


@transfer_to_logical_device.override(Tensor)
def transfer_to_logical_device_default(tensor: Tensor, ordinal: int):
    return shark_turbine.ops.iree.transfer_to_logical_device(
        f"{ordinal}", unbox_tensor(tensor)
    )


# Sharded default impls (do nothing).


@sharded_cat.override(Tensor)
def sharded_cat_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)


@sharded_sum.override(Tensor)
def sharded_sum_unsharded(maybe_sharded):
    return unbox_tensor(maybe_sharded)
