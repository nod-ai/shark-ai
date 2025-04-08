# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Optional, Sequence, Union, List, Tuple

import torch
import numbers
from torch import Tensor, dtype

from ..types import (
    AnyTensor,
    ShardedTensor,
    Theta,
    sharding,
    InferenceTensor,
    PrimitiveTensor,
)
from numbers import Number
import math

from ._registry import *

__all__ = [
    "all_gather",
    "all_reduce",
    "barrier_on_logical_device",
    "cat",
    "conv2d",
    "einsum_2args",
    "elementwise",
    "embedding_lookup",
    "equal",
    "expand",
    "flatten",
    "gather",
    "gelu_sigmoid_approximation",
    "gelu_tanh_approximation",
    "get_index",
    "gemm",
    "group_norm_affine",
    "layer_norm",
    "index_copy_",
    "index_put_",
    "index_select",
    "interpolate",
    "linear",
    "matmul",
    "mean",
    "module_register_buffer",
    "permute",
    "rms_norm",
    "repeat",
    "replicate",
    "reshape",
    "reshard",
    "reshard_split",
    "reshard_like",
    "scaled_dot_product_attention",
    "sharded_cat",
    "sharded_sum",
    "softmax",
    "squeeze",
    "to",
    "topk",
    "trace_tensor",
    "transfer_to_logical_device",
    "transpose",
    "unflatten",
    "unshard",
    "unsqueeze",
    "view",
    "view_as_complex",
    "view_as_real",
]

IntOrSequenceInt = Union[int, Sequence[int]]


@overridable
def all_gather(maybe_sharded: AnyTensor, *, dim: int | None = None) -> AnyTensor:
    "Gather/concatenate on all devices along dimension `dim`."
    ...


@all_gather.trampoline
def _all_gather_trampoline(
    d: SignatureDispatcher, maybe_sharded: AnyTensor, *, dim: int | None = None
):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded, dim=dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def all_reduce(tensor: AnyTensor) -> AnyTensor:
    "Reduce on all devices."
    ...


@all_reduce.trampoline
def _all_reduce_trampoline(d: SignatureDispatcher, tensor: AnyTensor):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def cat(tensors: Tuple[AnyTensor, ...] | List[AnyTensor], dim: int = 0) -> AnyTensor:
    ...


@cat.trampoline
def _cat_trampoline(
    d: SignatureDispatcher, tensors: Tuple[Tensor, ...] | List[Tensor], dim: int = 0
):
    for override in d.find_overrides(tensors):
        result = override(tensors, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def conv2d(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: Optional[torch.dtype] = None,
):
    """Equivalent to torch.nn.functional.conv2d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv2d.trampoline
def _conv2d_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            accum_dtype=accum_dtype,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def einsum_2args(
    input0: AnyTensor,
    input1: AnyTensor,
    einsum_str: str,
    *,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Executes a given Einstein summation notation string on the provided tensors.

    Equivalent to:
    ```
    y = torch.einsum(einsum_str, input0, input1)
    ```
    """
    raise NotImplementedError


@einsum_2args.trampoline
def _einsum_trampoline(
    d: SignatureDispatcher, input0: AnyTensor, input1: AnyTensor, einsum_str: str
):
    tensors = (input0, input1)
    for override in d.find_overrides(tensors):
        result = override(input0, input1, einsum_str)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def elementwise(operator, *args, **kwargs) -> AnyTensor:
    """Applies an elementwise operator against arguments."""
    raise NotImplementedError


@elementwise.trampoline
def _elementwise_trampoline(d: SignatureDispatcher, operator, *args, **kwargs):
    tensors = []
    for a in args:
        if isinstance(a, (Tensor, InferenceTensor)):
            tensors.append(a)
        else:
            break
    for override in d.find_overrides(tensors):
        result = override(operator, *args, **kwargs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def embedding_lookup(
    input: AnyTensor, embedding_matrix: AnyTensor, dtype: Optional[dtype]
) -> AnyTensor:
    """Performs the equivalent of F.embedding(input, embedding_matrix).

    Note that the default algorithm will unquantize the embedding_matrix to
    do the lookup, which is inefficient. Specializations should decompose
    this as appropriate for quantized arithmetic.
    """
    raise NotImplementedError


@embedding_lookup.trampoline
def _embedding_lookup_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    embedding_matrix: AnyTensor,
    dtype: Optional[dtype],
):
    tensors = (input, embedding_matrix)
    for override in d.find_overrides(tensors):
        result = override(input, embedding_matrix, dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def equal(a: AnyTensor, b: AnyTensor) -> bool:
    """Compares 2 tensors for equality, such that they elements and dtype are equal.

    Overrides are matched first against both tensor types and failing that,
    then on just the first.
    Therefore, each first-only argument override must internally decide whether
    it can handle an equality check with an arbitrary b tensor.
    """
    ...


@equal.trampoline
def _equal_trampoline(d: SignatureDispatcher, a: AnyTensor, b: AnyTensor):
    # Try first more specific matching the 2 operands.
    tensors = (
        a,
        b,
    )
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result

    # Less specific. Try matching only the first operand.
    tensors = (a,)
    for override in d.find_overrides(tensors):
        result = override(a, b)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def expand(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.expand"""
    ...


@expand.trampoline
def _expand_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, shape: List[int]
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, shape)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def get_index(
    tensor: AnyTensor,
    key: slice,
) -> torch.Tensor:
    """Indexes the tensor using the key.

    Equivalent to:
    ```
    out = tensor[key]
    ```
    """
    raise NotImplementedError


@get_index.trampoline
def _get_index_trampoline(d: SignatureDispatcher, tensor: AnyTensor, key: slice):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, key)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def flatten(input: AnyTensor, start_dim: int = 0, end_dim: int = -1) -> AnyTensor:
    """See torch.flatten"""
    ...


@flatten.trampoline
def _flatten_trampoline(
    d: SignatureDispatcher, input: AnyTensor, start_dim: int = 0, end_dim: int = -1
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, start_dim, end_dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def gather(input: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.gather"""
    ...


@gather.trampoline
def _gather_trampoline(
    d: SignatureDispatcher, input: AnyTensor, dim: int, index: AnyTensor
) -> AnyTensor:
    dispatch_args = (
        input,
        index,
    )
    for override in d.find_overrides(dispatch_args):
        result = override(input, dim, index)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


def gelu_sigmoid_approximation(input: AnyTensor) -> AnyTensor:
    """Applies GELU approximation that is fast but somewhat inaccurate.
    See: https://github.com/hendrycks/GELUs
    """
    return input * elementwise(torch.sigmoid, 1.702 * input)


def gelu_tanh_approximation(input: AnyTensor) -> AnyTensor:
    """Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    Approximation with tanh"""
    return (
        0.5
        * input
        * (
            1.0
            + elementwise(
                torch.tanh,
                math.sqrt(2.0 / math.pi)
                * (input + 0.044715 * elementwise(torch.pow, input, 3.0)),
            )
        )
    )


@overridable
def gemm(
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor] = None,
    alpha: Optional[Union[Number, AnyTensor]] = None,
    beta: Optional[Union[Number, AnyTensor]] = None,
    transa: bool = False,
    transb: bool = False,
):
    """GEMM as defined by BLAS.
    `alpha*a*b + beta*c`
    If `c` is None it is the zero-filed tensor.
    """
    raise NotImplementedError


@gemm.trampoline
def _gemm_trampoline(
    d: SignatureDispatcher,
    a: AnyTensor,
    b: AnyTensor,
    c: Optional[AnyTensor] = None,
    alpha: Optional[Union[Number, AnyTensor]] = None,
    beta: Optional[Union[Number, AnyTensor]] = None,
    transa: bool = False,
    transb: bool = False,
):
    tensors = (a, b, c)
    for override in d.find_overrides(tensors):
        result = override(
            a=a, b=b, c=c, alpha=alpha, beta=beta, transa=transa, transb=transb
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def group_norm_affine(
    input: AnyTensor, weight: AnyTensor, bias: AnyTensor, *, num_groups: int, eps: float
):
    """Equivalent to torch.nn.functional.group_norm(affine=True)."""
    raise NotImplementedError


@group_norm_affine.trampoline
def _group_norm_affine_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: AnyTensor,
    *,
    num_groups: int,
    eps: float,
):
    tensors = (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, num_groups=num_groups, eps=eps)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_copy_(
    inout: AnyTensor, dim: int, index: AnyTensor, tensor: AnyTensor
) -> AnyTensor:
    """See torch.Tensor.index_copy_"""
    ...


@index_copy_.trampoline
def _index_copy__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    tensor: AnyTensor,
) -> AnyTensor:
    tensors = (inout, index, tensor)
    for override in d.find_overrides(tensors):
        result = override(inout, dim, index, tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_put_(
    inout: AnyTensor, indices: Tuple[AnyTensor], values: AnyTensor
) -> AnyTensor:
    """See torch.Tensor.index_put_"""
    ...


@index_put_.trampoline
def _index_put__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    indices: Tuple[AnyTensor],
    values: AnyTensor,
) -> AnyTensor:
    # We change the order for the variadic indices to be last.
    tensors = (inout, values, *indices)
    for override in d.find_overrides(tensors):
        result = override(inout, indices, values)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def index_select(tensor: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.Tensor.index_select"""
    ...


@index_select.trampoline
def _index_select_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim: int, index: AnyTensor
) -> AnyTensor:
    tensors = (tensor, index)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim, index)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def interpolate(
    input: AnyTensor,
    size: Optional[int | List[int]] = None,
    scale_factor: Optional[float | List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> AnyTensor:
    """Equivalent to torch.nn.functional.interpolate"""
    raise NotImplementedError


@interpolate.trampoline
def _interpolate_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    size: Optional[int | List[int]] = None,
    scale_factor: Optional[float | List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> AnyTensor:
    tensors = [input]
    for override in d.find_overrides(tensors):
        result = override(
            input,
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            antialias,
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def layer_norm(
    input: AnyTensor,
    weight: Optional[AnyTensor],
    bias: Optional[AnyTensor],
    *,
    eps: float,
    normalized_shape: Optional[tuple[int]] = None,
):
    """Equivalent to torch.nn.functional.layer_norm(elementwise_affine=True)."""
    raise NotImplementedError


@layer_norm.trampoline
def _layer_norm_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: Optional[AnyTensor],
    bias: Optional[AnyTensor],
    *,
    eps: float,
    normalized_shape: Optional[tuple[int]] = None,
):
    tensors = [input]
    if weight is not None:
        tensors.append(bias)
    if bias is not None:
        tensors.append(bias)
    for override in d.find_overrides(tensors):
        result = override(
            input, weight, bias, eps=eps, normalized_shape=normalized_shape
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def linear(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data.

    Equivalent to:
    ```
    y = torch.matmul(input, weight.T) + bias
    ```

    This operator is defined to operate on a limited number of quantized types.
    In that situation, the result may be a QuantizedTensor. Callers should
    be prepared to handle this scenario.

    The optional accum_dtype argument is used as a hint to some implementations
    which may need help in selecting an appropriate high precision type for
    accumulation.
    """
    raise NotImplementedError


@linear.trampoline
def _linear_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
):
    tensors = (input, weight) if bias is None else (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(input, weight, bias, accum_dtype=accum_dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def matmul(lhs: AnyTensor, rhs: AnyTensor, *, transpose_rhs: bool = False):
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.T)` and the `transpose_rhs=` defaults to True, indicating
    the the RHS is expected to have been transposed already (by some outside
    force). Most inference optimizers will store their weights in this way
    and assume fusions that operate on them, so we just make it the default.

    Args:
    lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
    rhs: Right hand side tensor. Must be 2d or a scalar.
    transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    raise NotImplementedError


@matmul.trampoline
def _matmul_trampoline(
    d: SignatureDispatcher, lhs, rhs, *, transpose_rhs: bool = False
):
    tensors = (lhs, rhs)
    for override in d.find_overrides(tensors):
        result = override(lhs, rhs, transpose_rhs=transpose_rhs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def permute(tensor: AnyTensor, dims: List[int]) -> AnyTensor:
    """Permute the tensor dimensions according to the permutation `dims` in line
    notation.
    The semantics are the same as torch.permute."""
    ...


@permute.trampoline
def _permute_trampoline(d: SignatureDispatcher, tensor: AnyTensor, dims: List[int]):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dims)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def mean(
    x: AnyTensor,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    """See torch.mean"""
    raise NotImplementedError


@mean.trampoline
def _mean_trampoline(
    d: SignatureDispatcher,
    x: AnyTensor,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    tensors = (x,)
    for override in d.find_overrides(tensors):
        result = override(x, dim, keepdim, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def module_register_buffer(
    module: torch.nn.Module, name: str, tensor: AnyTensor
) -> None:
    """Register the tensor into the module. See torch.nn.Module.register_buffer."""
    ...


@module_register_buffer.trampoline
def _module_register_buffer_trampoline(
    d: SignatureDispatcher, module: torch.nn.Module, name: str, tensor: AnyTensor
) -> None:
    args = (module, tensor)
    for override in d.find_overrides(args):
        result = override(module, name, tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(args)


@overridable
def rms_norm(
    x: AnyTensor, weight: AnyTensor, *, epsilon: float, orig_dtype: torch.dtype
) -> AnyTensor:
    """Computes the full, unbiased RMS normalization of an input."""
    raise NotImplementedError


@rms_norm.trampoline
def _rms_norm_trampoline(
    d: SignatureDispatcher,
    x: AnyTensor,
    weight: AnyTensor,
    *,
    epsilon: float,
    orig_dtype: torch.dtype,
):
    tensors = (x, weight)
    for override in d.find_overrides(tensors):
        result = override(x, weight, epsilon=epsilon, orig_dtype=orig_dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def repeat(input: AnyTensor, *sizes: List[int]) -> AnyTensor:
    """See torch.Tensor.repeat"""
    ...


@repeat.trampoline
def _repeat_trampoline(
    d: SignatureDispatcher, input: AnyTensor, *sizes: List[int]
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, *sizes)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def replicate(
    input: AnyTensor, count: int, devices: tuple[int, ...] | None
) -> ShardedTensor:
    """Replicate across devices.

    Possibly reshards if required."""
    ...


@replicate.trampoline
def _replicate_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    count: int,
    devices: tuple[int, ...] | None = None,
) -> ShardedTensor:
    tensors = (input,)
    if isinstance(input, (torch.Tensor, PrimitiveTensor)):
        devices = devices if devices is not None else tuple(range(count))
    else:
        assert devices is None

    for override in d.find_overrides(tensors):
        result = override(input, count=count, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def scaled_dot_product_attention(
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    a: Optional[AnyTensor],
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> AnyTensor:
    """Computes the scaled dot product attention using QKV."""
    raise NotImplementedError


@scaled_dot_product_attention.trampoline
def _scaled_dot_product_attention(
    d: SignatureDispatcher,
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    a: Optional[AnyTensor],
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    tensors = (q, k, v, a)
    for override in d.find_overrides(tensors):
        if is_causal is not None:
            result = override(q, k, v, a, is_causal=is_causal, scale=scale)
        else:
            result = override(q, k, v, a, scale=scale)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def reshape(input: AnyTensor, shape: List[int]) -> AnyTensor:
    """Returns a tensor with the same data and number of elements as input, but with
    the specified shape.
    See torch.reshape.
    """
    ...


@reshape.trampoline
def _reshape_trampoline(d: SignatureDispatcher, input, shape) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, shape)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def reshard(
    input: AnyTensor | Theta,
    spec: (
        sharding.TensorSharding | sharding.ThetaLayerSharding | sharding.ThetaSharding
    ),
) -> AnyTensor | Theta:
    """Reshard to the given specification.
    If a Theta is given then the tensor nesting is preserved,
    but the tensors are sharded according to the spec.
    """
    ...


@reshard.trampoline
def _reshard_trampoline(d: SignatureDispatcher, input, spec) -> ShardedTensor:
    dispatch_args = (input, spec)
    for override in d.find_overrides(dispatch_args):
        result = override(input, spec)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def reshard_split(
    input: AnyTensor, *, dim: int, count: int, devices: tuple[int, ...] | None
) -> ShardedTensor:
    """Split `input` along `dim`.
    This does not mean that a sharded tensor is further sharded.
    It is not composition of sharding operations.
    """
    ...


@reshard_split.trampoline
def _reshard_split_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    dim: int,
    count: int,
    devices: tuple[int, ...] | None = None,
) -> ShardedTensor:
    tensors = (input,)
    if isinstance(input, (torch.Tensor, PrimitiveTensor)):
        devices = devices if devices is not None else tuple(range(count))
    else:
        assert devices is None

    for override in d.find_overrides(tensors):
        result = override(input, dim=dim, count=count, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def reshard_like(input: AnyTensor, like: AnyTensor) -> AnyTensor:
    """Shard `input` the same way as `like`.

    This may require expensive resharding."""
    ...


@reshard_like.trampoline
def _reshard_like_trampoline(
    d: SignatureDispatcher, input: AnyTensor, like: AnyTensor
) -> AnyTensor:
    tensors = (
        input,
        like,
    )
    for override in d.find_overrides(tensors):
        result = override(input, like)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sharded_cat(maybe_sharded: AnyTensor):
    """Concats all shards along the sharding dimension.

    Does nothing if not sharded.
    """
    raise NotImplementedError


@sharded_cat.trampoline
def _sharded_cat_trampoline(d: SignatureDispatcher, maybe_sharded: AnyTensor):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def sharded_sum(maybe_sharded: AnyTensor):
    ...


@sharded_sum.trampoline
def _sharded_sum_trampoline(d: SignatureDispatcher, maybe_sharded: AnyTensor):
    tensors = (maybe_sharded,)
    for override in d.find_overrides(tensors):
        result = override(maybe_sharded)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def softmax(
    tensor: AnyTensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None
) -> AnyTensor:
    """See torch.nn.functional.softmax"""
    ...


@softmax.trampoline
def _softmax_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> AnyTensor:
    dispatch_args = [tensor]
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, dim=dim, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def to(tensor: AnyTensor, *args, **kwargs) -> AnyTensor:
    """See torch.Tensor.to"""
    ...


@to.trampoline
def _to_trampoline(d: SignatureDispatcher, tensor: AnyTensor, *args, **kwargs):
    dispatch_args = [tensor]
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, *args, **kwargs)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def trace_tensor(key: str, *tensors: tuple[AnyTensor]):
    ...


@trace_tensor.trampoline
def _transfer_to_logical_device_trampoline(
    d: SignatureDispatcher, key: str, *tensors: tuple[AnyTensor]
):
    for override in d.find_overrides(tensors):
        result = override(key, *tensors)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def barrier_on_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@barrier_on_logical_device.trampoline
def _barrier_on_logical_device_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, ordinal: int
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, ordinal)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def transfer_to_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@transfer_to_logical_device.trampoline
def _transfer_to_logical_device_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, ordinal: int
):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, ordinal)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def transpose(tensor: AnyTensor, dim0: int, dim1: int) -> AnyTensor:
    """See torch.transpose"""
    ...


@transpose.trampoline
def _transpose_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim0: int, dim1: int
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim0, dim1)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def unflatten(input: AnyTensor, dim: int, sizes: Tuple[int]) -> AnyTensor:
    """See torch.unflatten"""
    ...


@unflatten.trampoline
def _unflatten_trampoline(
    d: SignatureDispatcher, input: AnyTensor, dim: int, sizes: Tuple[int]
) -> AnyTensor:
    dispatch_args = (input,)
    for override in d.find_overrides(dispatch_args):
        result = override(input, dim, sizes)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable
def unshard(tensor: AnyTensor) -> AnyTensor:
    """Return the tensor that has the same elements and shape, but is not sharded."""
    ...


@unshard.trampoline
def _unshard_trampoline(d: SignatureDispatcher, tensor: AnyTensor):
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def unsqueeze(tensor: AnyTensor, dim: int) -> AnyTensor:
    """See torch.unsqueeze"""
    ...


@unsqueeze.trampoline
def _unsqueeze_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, dim: int
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def squeeze(tensor, dim: Optional[int]) -> AnyTensor:
    """See torch.squeeze"""
    ...


@squeeze.trampoline
def _squeeze_trampoline(
    d: SignatureDispatcher, tensor, dim: Optional[int]
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensor):
        result = override(tensor, dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def topk(tensor, k: int, dim: int) -> AnyTensor:
    """See torch.topk"""
    ...


@topk.trampoline
def _topk_trampoline(d: SignatureDispatcher, tensor, k: int, dim: int) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensor):
        result = override(tensor, k=k, dim=dim)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.view"""
    ...


@view.trampoline
def _view_trampoline(
    d: SignatureDispatcher, tensor: AnyTensor, shape: List[int]
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor, shape)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view_as_complex(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.view_as_complex"""
    ...


@view_as_complex.trampoline
def _view_as_complex_trampoline(d: SignatureDispatcher, tensor: AnyTensor) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def view_as_real(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.view_as_complex"""
    ...


@view_as_real.trampoline
def _view_as_real_trampoline(d: SignatureDispatcher, tensor: AnyTensor) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        result = override(tensor)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)
