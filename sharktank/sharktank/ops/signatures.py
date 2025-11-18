# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Optional, Sequence, Union, List, Tuple
from numbers import Number, Integral
import math

import torch
from torch import Tensor, dtype

from amdsharktank.types import (
    AnyTensor,
    BlockScaledPackedLayout,
    QuantizedLayout,
    QuantizerTensor,
    Slice,
    ShardedTensor,
    SplitPrimitiveTensor,
    Theta,
    sharding,
    InferenceTensor,
    PrimitiveTensor,
    UnnamedTensorName,
)


from ._registry import *

__all__ = [
    "abs",
    "all_gather",
    "all_reduce",
    "arange",
    "argmax",
    "barrier_on_logical_device",
    "cat",
    "chunk",
    "conv2d",
    "conv3d",
    "conv1d",
    "cos",
    "dequantize",
    "einsum_2args",
    "elementwise",
    "embedding_lookup",
    "equal",
    "expand",
    "extend_attention",
    "extract_slice",
    "flatten",
    "full",
    "gather",
    "gelu_sigmoid_approximation",
    "gelu_tanh_approximation",
    "gemm",
    "group_norm_affine",
    "layer_norm",
    "log",
    "logical_or",
    "index_copy_",
    "index_put_",
    "index_select",
    "interpolate",
    "linear",
    "masked_fill",
    "matmul",
    "mean",
    "module_register_buffer",
    "ones",
    "ones_like",
    "pad",
    "permute",
    "quantize",
    "rms_norm",
    "reduce_scatter",
    "repeat",
    "replicate",
    "reshape",
    "reshard",
    "reshard_split",
    "reshard_like",
    "scaled_dot_product_attention",
    "scatter_",
    "scatter_add",
    "sharded_cat",
    "sharded_sum",
    "sharded_gather",
    "shards",
    "sigmoid",
    "sin",
    "softmax",
    "split",
    "squeeze",
    "sum",
    "swiglu",
    "tensor",
    "to",
    "topk",
    "trace_tensor",
    "transfer_to_logical_device",
    "transpose",
    "unflatten",
    "unpack",
    "unpack_qs",
    "unpack_to_qs",
    "unshard",
    "unsqueeze",
    "view",
    "view_as_complex",
    "view_as_real",
    "where",
    "zeros",
    "zeros_like",
]

IntOrSequenceInt = Union[int, Sequence[int]]


@overridable(dispatch_args=(0,))
def abs(tensor: AnyTensor) -> AnyTensor:
    """See torch.abs"""
    ...


@overridable(is_trivially_replicable=False)
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


@overridable(is_trivially_replicable=False)
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


@overridable(dispatch_args=(), is_trivially_replicable=False)
def arange(
    *start_end_step,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    """
    See torch.arange.

    If devices is given, returns a ReplicatedTensor with identical (but independently created) shards.
    """
    ...


@overridable(dispatch_args=("tensor",))
def argmax(
    tensor: AnyTensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
) -> AnyTensor:
    "Take argmax of the tensor"
    ...


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


@overridable(dispatch_args=(0,))
def chunk(tensor: AnyTensor, chunks: int, dim: int = 0) -> tuple[AnyTensor, ...]:
    """See torch.chunk"""
    ...


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
def conv3d(
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
    """Equivalent to torch.nn.functional.conv3d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv3d.trampoline
def _conv3d_trampoline(
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
def conv1d(
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
    """Equivalent to torch.nn.functional.conv1d with enhancements:

    * Primitive weight/bias tensors will be promoted to the input dtype.
    """
    raise NotImplementedError


@conv1d.trampoline
def _conv1d_trampoline(
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


@overridable(dispatch_args=(0,))
def cos(tensor: AnyTensor) -> AnyTensor:
    """See torch.cos"""
    ...


@overridable
def dequantize(
    input: AnyTensor | QuantizedLayout | dict[str, AnyTensor],
    /,
    *,
    quantizer: AnyTensor | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTensor:
    """Dequantize a tensor. The input may be a quantized tensor, layout or a
    dictionary of planes.

    In some cases it is allowed for a plane to be missing if a quantizer is given.
    E.g. when we have a StaticScaledQuantizer the scale plane is not required."""
    ...


@dequantize.trampoline
def _dequantize_trampoline(
    d: SignatureDispatcher,
    input: AnyTensor,
    /,
    *,
    quantizer: AnyTensor | None = None,
    dtype: torch.dtype | None = None,
) -> AnyTensor:
    dispatch_args = (input, quantizer)
    for override in d.find_overrides(dispatch_args):
        result = override(input, quantizer=quantizer, dtype=dtype)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


@overridable(dispatch_args=(0, 1))
def einsum_2args(
    input0: AnyTensor,
    input1: AnyTensor,
    einsum_str: str,
) -> torch.Tensor:
    """Executes a given Einstein summation notation string on the provided tensors.

    Equivalent to:
    ```
    y = torch.einsum(einsum_str, input0, input1)
    ```
    """
    raise NotImplementedError


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


@overridable(dispatch_args=(0, 1))
def embedding_lookup(
    input: AnyTensor, embedding_matrix: AnyTensor, dtype: Optional[dtype]
) -> AnyTensor:
    """Performs the equivalent of F.embedding(input, embedding_matrix).

    Note that the default algorithm will unquantize the embedding_matrix to
    do the lookup, which is inefficient. Specializations should decompose
    this as appropriate for quantized arithmetic.
    """
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def empty_like(
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> AnyTensor:
    """See torch.zeros_like"""
    ...


@overridable(is_trivially_replicable=False)
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


@overridable(dispatch_args=(0,))
def expand(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.expand"""
    ...


@overridable(dispatch_args=(0,))
def extract_slice(
    tensor: AnyTensor,
    key: Slice,
) -> torch.Tensor:
    """Indexes the tensor using the key.

    Equivalent to:
    ```
    out = tensor[key]
    ```
    """
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def flatten(input: AnyTensor, start_dim: int = 0, end_dim: int = -1) -> AnyTensor:
    """See torch.flatten"""
    ...


@overridable(dispatch_args=(), is_trivially_replicable=False)
def full(
    size: Sequence[int],
    fill_value: Number,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    """
    See torch.full.

    If devices is given, returns a ReplicatedTensor with identical (but independently created) shards.
    """
    ...


@overridable(dispatch_args=("input", "index"))
def gather(input: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.gather"""
    ...


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


@overridable(dispatch_args=("a", "b", "c"))
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


@overridable(dispatch_args=("input", "weight", "bias"))
def group_norm_affine(
    input: AnyTensor, weight: AnyTensor, bias: AnyTensor, *, num_groups: int, eps: float
):
    """Equivalent to torch.nn.functional.group_norm(affine=True)."""
    raise NotImplementedError


@overridable(dispatch_args=("inout", "index", "tensor"))
def index_copy_(
    inout: AnyTensor, dim: int, index: AnyTensor, tensor: AnyTensor
) -> AnyTensor:
    """See torch.Tensor.index_copy_"""
    ...


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


@overridable(dispatch_args=("tensor", "index"))
def index_select(tensor: AnyTensor, dim: int, index: AnyTensor) -> AnyTensor:
    """See torch.Tensor.index_select"""
    ...


@overridable(dispatch_args=(0,))
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


@overridable(dispatch_args=(0,))
def log(tensor: AnyTensor) -> AnyTensor:
    """See torch.log"""
    ...


@overridable(dispatch_args=(0, 1))
def logical_or(input: AnyTensor, other: AnyTensor) -> AnyTensor:
    """See torch.logical_or"""
    ...


@overridable
def linear(
    input: AnyTensor,
    weight: AnyTensor,
    bias: Optional[AnyTensor] = None,
    *,
    accum_dtype: Optional[torch.dtype] = None,
    matmul_impl: Optional[str] = None,
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data.

    Equivalent to:
    ```
    y = torch.matmul(input, weight.mT) + bias
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
    matmul_impl: Optional[str] = None,
):
    tensors = (input, weight) if bias is None else (input, weight, bias)
    for override in d.find_overrides(tensors):
        result = override(
            input, weight, bias, accum_dtype=accum_dtype, matmul_impl=matmul_impl
        )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(dispatch_args=(0, 1))
def masked_fill(input: AnyTensor, mask: AnyTensor, value: Number) -> AnyTensor:
    """See torch.masked_fill"""
    ...


@overridable(dispatch_args=(0, 1))
def matmul(lhs: AnyTensor, rhs: AnyTensor, *, transpose_rhs: bool = False):
    """Performs a matmul where the RHS may be an InferenceTensor.

    Unlike torch.matmul, this variant is optimized for emission of a fused
    `matmul(lhs, rhs.mT)` when `transpose_rhs=True`. Most inference optimizers
    will store their weights in this way and assume fusions that operate on them.

    Args:
    lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
    rhs: Right hand side tensor. Must be 2d or a scalar.
    transpose_rhs: Whether the right hand side should be transposed prior
        to matmul.
    """
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def pad(
    input: AnyTensor,
    _pad: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> AnyTensor:
    """See torch.nn.functional.pad"""
    ...


@overridable(dispatch_args=(0,))
def permute(tensor: AnyTensor, dims: List[int]) -> AnyTensor:
    """Permute the tensor dimensions according to the permutation `dims` in line
    notation.
    The semantics are the same as torch.permute."""
    ...


@overridable(dispatch_args=(0,))
def mean(
    x: AnyTensor,
    dim: Union[int, List[int]],
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    """See torch.mean"""
    raise NotImplementedError


@overridable(dispatch_args=("module", "tensor"), is_trivially_replicable=False)
def module_register_buffer(
    module: torch.nn.Module, name: str, tensor: AnyTensor
) -> None:
    """Register the tensor into the module. See torch.nn.Module.register_buffer."""
    ...


@overridable(dispatch_args=(), is_trivially_replicable=False)
def ones(
    *size,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    """
    See torch.ones.

    If devices is given, returns a ReplicatedTensor with identical (but independently created) shards.
    """
    ...


@overridable(dispatch_args=(0,))
def ones_like(
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> AnyTensor:
    """See torch.ones_like"""
    ...


@overridable(dispatch_args=(0, 1))
def quantize(
    tensor: AnyTensor, quantizer: AnyTensor, name: str = UnnamedTensorName
) -> AnyTensor:
    """Quantize a tensor using the provided quantizer."""
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def reduce_scatter(tensor: AnyTensor, scatter_dim: int) -> AnyTensor:
    """Reduces then splits/scatters across the devices."""
    ...


@overridable(dispatch_args=(0, 1))
def rms_norm(
    x: AnyTensor, weight: AnyTensor, *, epsilon: float, orig_dtype: torch.dtype
) -> AnyTensor:
    """Computes the full, unbiased RMS normalization of an input."""
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def repeat(input: AnyTensor, *sizes: List[int]) -> AnyTensor:
    """See torch.Tensor.repeat"""
    ...


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
    if isinstance(input, ShardedTensor):
        assert devices is None
    else:
        devices = devices if devices is not None else tuple(range(count))

    for override in d.find_overrides(tensors):
        result = override(input, count=count, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(dispatch_args=("q", "k", "v", "a"))
def scaled_dot_product_attention(
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    a: Optional[AnyTensor],
    sink: Optional[AnyTensor] = None,
    sliding_window: Optional[AnyTensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    *,
    impl: Optional[str] = None,
) -> AnyTensor:
    """Computes the scaled dot product attention using QKV."""
    raise NotImplementedError


@overridable(dispatch_args=("q", "k", "v"))
def extend_attention(
    q: AnyTensor,
    k: AnyTensor,
    v: AnyTensor,
    kv_cache: Optional[AnyTensor] = None,
    page_ids: Optional[AnyTensor] = None,
    start_positions: Optional[AnyTensor] = None,
    seq_lens: Optional[AnyTensor] = None,
    *,
    impl: Optional[str] = None,
) -> AnyTensor:
    """Computes the extend attention using QKV."""
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def reshape(input: AnyTensor, shape: List[int]) -> AnyTensor:
    """Returns a tensor with the same data and number of elements as input, but with
    the specified shape.
    See torch.reshape.
    """
    ...


@overridable(dispatch_args=(0, 1), is_trivially_replicable=False)
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


@overridable(is_trivially_replicable=False)
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


@overridable(dispatch_args=("input", "like"), is_trivially_replicable=False)
def reshard_like(input: AnyTensor, like: AnyTensor) -> AnyTensor:
    """Shard `input` the same way as `like`.

    This may require expensive resharding."""
    ...


@overridable(dispatch_args=("inout", "index", "src"))
def scatter_(
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    src: AnyTensor | Number,
    *,
    reduce: str = None,
):
    """
    See torch.Tensor.scatter_
    NOTE: Does not modify the inout tensor in place for ShardedTensors, will return copy.
    """
    ...


@overridable(dispatch_args=("input", "index", "src"))
def scatter_add(
    input: AnyTensor, dim: int, index: AnyTensor, src: AnyTensor
) -> AnyTensor:
    """
    See torch.scatter_add
    """
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def sharded_cat(maybe_sharded: AnyTensor):
    """Concats all shards along the sharding dimension.

    Does nothing if not sharded.
    """
    raise NotImplementedError


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def sharded_gather(input: AnyTensor, root_rank: int) -> list[AnyTensor]:
    """Gather the input tensor from all devices to the given device ordinal."""
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def shards(input: ShardedTensor | QuantizedLayout) -> list[AnyTensor | QuantizedLayout]:
    """Return the shards of a sharded tensor."""
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def sharded_sum(maybe_sharded: AnyTensor, root_rank: int = 0) -> AnyTensor:
    """Reduce across the shards into a single device.

    root_rank:
        Rank of receiving device within the tensor devices.
        If sharded, `maybe_sharded.devices[root_rank]` is the destination.
    """
    ...


@overridable(dispatch_args=(0,))
def sigmoid(tensor: AnyTensor) -> AnyTensor:
    """See torch.sigmoid"""
    ...


@overridable(dispatch_args=(0,))
def sin(tensor: AnyTensor) -> AnyTensor:
    """See torch.sin"""
    ...


@overridable(dispatch_args=(0,))
def softmax(
    tensor: AnyTensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None
) -> AnyTensor:
    """See torch.nn.functional.softmax"""
    ...


@overridable(dispatch_args=(0,))
def split(
    tensor: AnyTensor, split_size_or_sections: int | list[int], dim: int = 0
) -> tuple[AnyTensor, ...]:
    """See torch.split"""
    ...


@overridable(dispatch_args=(0,))
def swiglu(
    tensor: AnyTensor, *, alpha: float = 1.702, limit: float | None = None
) -> AnyTensor:
    raise NotImplementedError


@overridable(dispatch_args=(0,))
def to(tensor: AnyTensor, *args, **kwargs) -> AnyTensor:
    """See torch.Tensor.to"""
    ...


@overridable(dispatch_args=("tensors",))
def trace_tensor(key: str, *tensors: tuple[AnyTensor, ...]):
    """Trace tensor(s) in IREE runtime or in eager mode.

    You can add trace_tensor into your model wherever you want. It will insert a
    trace op into the IR. Then you can register a callback in the IREE runtime for
    custom handling of the trace command during execution. For example recording the
    tensor into a file. There is also a destination/sink for eager execution.

    The trace op will prevent fusion which will influence how the model is compiled.
    This may change the behavior of the program and cause a numerical issue to
    disappear if it was the result of op fusion.

    Example usage at amdsharktank/tests/ops/ops_test.py::TestTraceTensors.

    See:
    amdsharktank.utils.debugging.set_trace_tensor_callback
    amdsharktank.utils.debugging.trace_tensor_to_safetensors_callback
    amdsharktank.utils.debugging.flags.trace_path
    amdsharktank.utils.iree.make_hal_buffer_view_trace_default_callback
    amdsharktank.layers.BaseLayer.trace_tensor
    """
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def barrier_on_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def transfer_to_logical_device(tensor: AnyTensor, ordinal: int) -> AnyTensor:
    """Transfer the tensor to a device with ordinal `ordinal`."""
    ...


@overridable(dispatch_args=(0,))
def transpose(tensor: AnyTensor, dim0: int, dim1: int) -> AnyTensor:
    """See torch.transpose"""
    ...


@overridable(dispatch_args=(0,))
def unflatten(input: AnyTensor, dim: int, sizes: Tuple[int]) -> AnyTensor:
    """See torch.unflatten"""
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def unpack(input: AnyTensor) -> QuantizedLayout:
    ...


@overridable(dispatch_args=(0, 1))
def unpack_qs(qs: AnyTensor, layout: BlockScaledPackedLayout) -> AnyTensor:
    """Return the unpacked unscaled/quantized values of a block scales packed layout."""
    ...


@overridable(dispatch_args=(0,))
def unpack_to_qs(input: AnyTensor) -> AnyTensor:
    ...


@overridable(dispatch_args=(0,), is_trivially_replicable=False)
def unshard(tensor: AnyTensor) -> AnyTensor:
    """Return the tensor that has the same elements and shape, but is not sharded."""
    ...


@overridable(dispatch_args=(0,))
def unsqueeze(tensor: AnyTensor, dim: int) -> AnyTensor:
    """See torch.unsqueeze"""
    ...


@overridable(dispatch_args=(0,))
def squeeze(tensor, dim: Optional[int]) -> AnyTensor:
    """See torch.squeeze"""
    ...


@overridable(dispatch_args=(0,))
def sum(
    input: AnyTensor,
    dim: int | List[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype = None,
) -> AnyTensor:
    """See torch.sum"""
    ...


@overridable()
def tensor(
    data: AnyTensor | Number,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    """See torch.tensor"""
    ...


@tensor.trampoline
def _tensor_trampoline(
    d: SignatureDispatcher,
    data: AnyTensor | Number,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    tensors = (data,) if isinstance(data, AnyTensor) else tuple()
    for override in d.find_overrides(tensors):
        result = override(data, dtype=dtype, device=device, devices=devices)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable
def topk(
    tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    chunk_size: Optional[int] = None,
    use_linalgext_topk: bool = False,
) -> AnyTensor:
    """See torch.topk"""
    ...


@topk.trampoline
def _topk_trampoline(
    d: SignatureDispatcher,
    tensor,
    k: int,
    dim: int,
    largest: bool = True,
    sorted: bool = True,
    chunk_size: Optional[int] = None,
    use_linalgext_topk: bool = False,
) -> AnyTensor:
    tensors = (tensor,)
    for override in d.find_overrides(tensors):
        if isinstance(tensor, SplitPrimitiveTensor):
            result = override(
                tensor,
                k=k,
                dim=dim,
                largest=largest,
                sorted=sorted,
                use_linalgext_topk=use_linalgext_topk,
            )

        else:
            result = override(
                tensor,
                k=k,
                dim=dim,
                largest=largest,
                sorted=sorted,
                chunk_size=chunk_size,
                use_linalgext_topk=use_linalgext_topk,
            )
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(dispatch_args=(0,))
def view(
    tensor: AnyTensor, shape: List[int] | None = None, dtype: torch.dtype | None = None
) -> AnyTensor:
    """See torch.Tensor.view"""
    ...


@overridable(dispatch_args=(0,))
def view_as_complex(tensor: AnyTensor) -> AnyTensor:
    """See torch.Tensor.view_as_complex"""
    ...


@overridable(dispatch_args=(0,))
def view_as_real(tensor: AnyTensor) -> AnyTensor:
    """See torch.Tensor.view_as_real"""
    ...


@overridable()
def where(
    condition: AnyTensor,
    input: AnyTensor | Number | None = None,
    other: AnyTensor | Number | None = None,
) -> AnyTensor | Tuple[AnyTensor, ...]:
    """
    See torch.where.

    If devices is given, returns a ReplicatedTensor with identical (but independently created) shards.
    """
    ...


@where.trampoline
def _where_trampoline(
    d: SignatureDispatcher,
    condition: AnyTensor,
    input: AnyTensor | Number | None = None,
    other: AnyTensor | Number | None = None,
):
    assert (input is None) == (other is None)
    tensors = [condition]
    if isinstance(input, AnyTensor):
        tensors.append(input)
    if isinstance(other, AnyTensor):
        tensors.append(other)
    tensors = tuple(tensors)

    for override in d.find_overrides(tensors):
        result = override(condition, input, other)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(tensors)


@overridable(dispatch_args=(), is_trivially_replicable=False)
def zeros(
    *size,
    dtype: Optional[dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    devices: Sequence[int] | None = None,
) -> AnyTensor:
    """
    See torch.zeros.

    If devices is given, returns a ReplicatedTensor with identical (but independently created) shards.
    """
    ...


@overridable(dispatch_args=(0,))
def zeros_like(
    tensor: AnyTensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> AnyTensor:
    """See torch.zeros_like"""
    ...
