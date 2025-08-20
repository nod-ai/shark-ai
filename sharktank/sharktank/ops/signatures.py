# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Optional, Sequence, Union, List, Tuple
from numbers import Number, Integral
import math
import inspect

import torch
from torch import Tensor, dtype

from sharktank.types import (
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
    "all_gather",
    "all_reduce",
    "argmax",
    "barrier_on_logical_device",
    "cat",
    "conv2d",
    "conv3d",
    "conv1d",
    "dequantize",
    "einsum_2args",
    "elementwise",
    "embedding_lookup",
    "equal",
    "expand",
    "extract_slice",
    "flatten",
    "gather",
    "gelu_sigmoid_approximation",
    "gelu_tanh_approximation",
    "gemm",
    "group_norm_affine",
    "layer_norm",
    "index_copy_",
    "index_put_",
    "index_select",
    "interpolate",
    "linear",
    "masked_fill",
    "matmul",
    "mean",
    "module_register_buffer",
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
    "softmax",
    "split",
    "squeeze",
    "sum",
    "swiglu",
    "to",
    "topk",
    "trace_tensor",
    "transfer_to_logical_device",
    "transpose",
    "unflatten",
    "unpack",
    "unpack_qs",
    "unshard",
    "unsqueeze",
    "view",
    "view_as_complex",
    "view_as_real",
    "zeros_like",
]

IntOrSequenceInt = Union[int, Sequence[int]]


def _call_override_with_defaults(override, args, kwargs, defaults=None):
    """Helper to call override function with proper argument mapping."""

    if defaults is None:
        defaults = {}

    # Use original function signature if wrapped (for sharding)
    func_to_inspect = (
        override._original_function
        if hasattr(override, "_original_function")
        else override
    )
    sig = inspect.signature(func_to_inspect)
    param_names = list(sig.parameters.keys())

    # Build final kwargs: start with defaults, override with user kwargs
    final_kwargs = defaults.copy()
    final_kwargs.update(kwargs)

    # Remove from final_kwargs any parameters that will be filled by positional args
    for i, param_name in enumerate(param_names):
        if i < len(args) and param_name in final_kwargs:
            del final_kwargs[param_name]

    return override(*args, **final_kwargs)


def create_overridable_op(
    name: str,
    is_trivially_replicable: bool = True,
    defaults: dict = None,
    dispatch_args: list[int] = None,
):
    """Factory that creates an overridable operation with generic trampoline.

    Args:
        name: Name of the operation
        is_trivially_replicable: Whether the operation is trivially replicable
        defaults: Default values for optional parameters
        dispatch_args: List of argument indices to use for dispatch. If None,
                      uses automatic discovery of all leading tensor arguments
    """

    @overridable(is_trivially_replicable=is_trivially_replicable)
    def op(*args, **kwargs):
        raise NotImplementedError(f"{name} not implemented")

    op.__name__ = name

    @op.trampoline
    def _trampoline(d: SignatureDispatcher, *args, **kwargs):
        if dispatch_args is not None:
            # Use explicitly specified dispatch arguments
            if len(set(dispatch_args)) != len(dispatch_args) or dispatch_args != sorted(
                dispatch_args
            ):
                raise ValueError("`dispatch_args` must be ordered and have no repeats")
            tensors = []
            op_sig = inspect.signature(op)
            param_names = list(op_sig.parameters.keys())

            for i in dispatch_args:
                if i < len(args):
                    tensors.append(args[i])
                else:
                    # Argument not provided - look up default value
                    default_value = None
                    if defaults and i < len(param_names):
                        param_name = param_names[i]
                        default_value = defaults.get(param_name, None)
                    tensors.append(default_value)
        else:
            # Use automatic discovery of all leading tensor arguments
            tensors = []
            for arg in args:
                if isinstance(arg, (Tensor, InferenceTensor)):
                    tensors.append(arg)
                elif (
                    isinstance(arg, (list, tuple))
                    and arg
                    and isinstance(arg[0], (Tensor, InferenceTensor))
                ):
                    # Handle collections like cat() - the collection itself is the dispatch arg
                    tensors = arg
                    break
                else:
                    # Stop at first non-tensor argument
                    break

        # Standard dispatch loop
        impl_selection = kwargs.get("impl")
        for override in d.find_overrides(tuple(tensors)):
            impl_name = getattr(override, "_impl_name", None)
            if impl_selection is not None:
                # All implementations should have a name for operations which use implementation selection
                assert impl_name is not None
                # Implementation selection will match on <example> but not <counterexample>
                if not impl_name.startswith(impl_selection):
                    continue

            result = _call_override_with_defaults(override, args, kwargs, defaults)
            if result is not NotImplemented:
                return override, result
        else:
            d.fail(tuple(tensors))

    return op


# Gather/concatenate on all devices along dimension `dim`.
# Args: maybe_sharded: AnyTensor, dim: Optional[int] = None
# Returns: AnyTensor
all_gather = create_overridable_op(
    "all_gather", is_trivially_replicable=False, defaults={"dim": None}
)


# Reduce on all devices.
# Args: tensor: AnyTensor
# Returns: AnyTensor
all_reduce = create_overridable_op("all_reduce", is_trivially_replicable=False)


# Take argmax of the tensor
# Args: tensor: AnyTensor, dim: Optional[int] = None,
#       keepdim: bool = False, chunk_size: Optional[int] = None
# Returns: AnyTensor
argmax = create_overridable_op(
    "argmax", defaults={"dim": None, "keepdim": False, "chunk_size": None}
)


cat = create_overridable_op("cat")


# Equivalent to torch.nn.functional.conv2d with enhancements:
# * Primitive weight/bias tensors will be promoted to the input dtype.
# Args: input: AnyTensor, weight: AnyTensor, bias: Optional[AnyTensor] = None,
#       stride: IntOrSequenceInt = 1, padding: IntOrSequenceInt = 0,
#       dilation: IntOrSequenceInt = 1, groups: IntOrSequenceInt = 1,
#       accum_dtype: Optional[torch.dtype] = None
conv2d = create_overridable_op(
    "conv2d",
    defaults={
        "bias": None,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "accum_dtype": None,
    },
)


# Equivalent to torch.nn.functional.conv3d with enhancements:
# * Primitive weight/bias tensors will be promoted to the input dtype.
# Args: input: AnyTensor, weight: AnyTensor, bias: Optional[AnyTensor] = None,
#       stride: IntOrSequenceInt = 1, padding: IntOrSequenceInt = 0,
#       dilation: IntOrSequenceInt = 1, groups: IntOrSequenceInt = 1,
#       accum_dtype: Optional[torch.dtype] = None
conv3d = create_overridable_op(
    "conv3d",
    defaults={
        "bias": None,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "accum_dtype": None,
    },
)


# Equivalent to torch.nn.functional.conv1d with enhancements:
# * Primitive weight/bias tensors will be promoted to the input dtype.
# Args: input: AnyTensor, weight: AnyTensor, bias: Optional[AnyTensor] = None,
#       stride: IntOrSequenceInt = 1, padding: IntOrSequenceInt = 0,
#       dilation: IntOrSequenceInt = 1, groups: IntOrSequenceInt = 1,
#       accum_dtype: Optional[torch.dtype] = None
conv1d = create_overridable_op(
    "conv1d",
    defaults={
        "bias": None,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "accum_dtype": None,
    },
)


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


# Executes a given Einstein summation notation string on the provided tensors.
# Equivalent to: y = torch.einsum(einsum_str, input0, input1)
# Args: input0: AnyTensor, input1: AnyTensor, einsum_str: str,
#       accum_dtype: Optional[torch.dtype] = None
# Returns: torch.Tensor
einsum_2args = create_overridable_op("einsum_2args", defaults={"accum_dtype": None})


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


# Performs the equivalent of F.embedding(input, embedding_matrix).
# Note that the default algorithm will unquantize the embedding_matrix to
# do the lookup, which is inefficient. Specializations should decompose
# this as appropriate for quantized arithmetic.
# Args: input: AnyTensor, embedding_matrix: AnyTensor, dtype: Optional[dtype]
# Returns: AnyTensor
embedding_lookup = create_overridable_op("embedding_lookup")


# See torch.empty_like
# Args: tensor: AnyTensor, dtype: Optional[torch.dtype] = None,
#       layout: Optional[torch.layout] = None, device: Optional[torch.device] = None,
#       requires_grad: bool = False, memory_format: torch.memory_format = torch.preserve_format
# Returns: AnyTensor
empty_like = create_overridable_op(
    "empty_like",
    defaults={
        "dtype": None,
        "layout": None,
        "device": None,
        "requires_grad": False,
        "memory_format": torch.preserve_format,
    },
)


# Compares 2 tensors for equality, such that they elements and dtype are equal.
# Overrides are matched first against both tensor types and failing that,
# then on just the first.
# Therefore, each first-only argument override must internally decide whether
# it can handle an equality check with an arbitrary b tensor.
# Args: a: AnyTensor, b: AnyTensor
# Returns: bool
equal = create_overridable_op("equal", is_trivially_replicable=False)


# See torch.Tensor.expand
# Args: tensor: AnyTensor, shape: List[int]
# Returns: AnyTensor
expand = create_overridable_op("expand")


# Indexes the tensor using the key.
# Equivalent to: out = tensor[key]
# Args: tensor: AnyTensor, key: Slice
# Returns: torch.Tensor
extract_slice = create_overridable_op("extract_slice")


# See torch.flatten
# Args: input: AnyTensor, start_dim: int = 0, end_dim: int = -1
# Returns: AnyTensor
flatten = create_overridable_op("flatten", defaults={"start_dim": 0, "end_dim": -1})


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


# TODO: Convert to factory function. The index tensor probably doesn't need to be part of the override.


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


# GEMM as defined by BLAS.
# `alpha*a*b + beta*c`
# If `c` is None it is the zero-filed tensor.
# Args: a: AnyTensor, b: AnyTensor, c: Optional[AnyTensor] = None,
#       alpha: Optional[Union[Number, AnyTensor]] = None,
#       beta: Optional[Union[Number, AnyTensor]] = None,
#       transa: bool = False, transb: bool = False
# Returns: AnyTensor
gemm = create_overridable_op(
    "gemm",
    defaults={"c": None, "alpha": None, "beta": None, "transa": False, "transb": False},
)


# Equivalent to torch.nn.functional.group_norm(affine=True).
# Args: input: AnyTensor, weight: AnyTensor, bias: AnyTensor,
#       num_groups: int, eps: float
# Returns: AnyTensor
group_norm_affine = create_overridable_op("group_norm_affine")


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


# See torch.Tensor.index_select
# Args: tensor: AnyTensor, dim: int, index: AnyTensor
# Returns: AnyTensor
index_select = create_overridable_op("index_select")


# Equivalent to torch.nn.functional.interpolate
# Args: input: AnyTensor, size: Optional[int | List[int]] = None,
#       scale_factor: Optional[float | List[float]] = None, mode: str = "nearest",
#       align_corners: Optional[bool] = None,
#       recompute_scale_factor: Optional[bool] = None, antialias: bool = False
# Returns: AnyTensor
interpolate = create_overridable_op(
    "interpolate",
    defaults={
        "size": None,
        "scale_factor": None,
        "mode": "nearest",
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False,
    },
)


# Equivalent to torch.nn.functional.layer_norm(elementwise_affine=True)
# Args: input: AnyTensor, weight: Optional[AnyTensor], bias: Optional[AnyTensor],
#       eps: float, normalized_shape: Optional[tuple[int]] = None
# Returns: AnyTensor
layer_norm = create_overridable_op("layer_norm", defaults={"normalized_shape": None})


# Applies a linear transformation to the incoming data.
# Equivalent to: y = torch.matmul(input, weight.mT) + bias
# This operator is defined to operate on a limited number of quantized types.
# In that situation, the result may be a QuantizedTensor. Callers should
# be prepared to handle this scenario.
# The optional accum_dtype argument is used as a hint to some implementations
# which may need help in selecting an appropriate high precision type for
# accumulation.
# Args: input: AnyTensor, weight: AnyTensor, bias: Optional[AnyTensor] = None,
#       accum_dtype: Optional[torch.dtype] = None
# Returns: torch.Tensor
linear = create_overridable_op("linear", defaults={"bias": None, "accum_dtype": None})


# See torch.masked_fill
# Args: input: AnyTensor, mask: AnyTensor, value: Number
# Returns: AnyTensor
masked_fill = create_overridable_op("masked_fill")


# Performs a matmul where the RHS may be an InferenceTensor.
# Unlike torch.matmul, this variant is optimized for emission of a fused
# `matmul(lhs, rhs.mT)` when `transpose_rhs=True`. Most inference optimizers
# will store their weights in this way and assume fusions that operate on them.
# Args: lhs: AnyTensor, rhs: AnyTensor, transpose_rhs: bool = False
# lhs: Left hand side tensor. Can have dimensionality > 2 for batch.
# rhs: Right hand side tensor. Must be 2d or a scalar.
# transpose_rhs: Whether the right hand side should be transposed prior to matmul.
matmul = create_overridable_op("matmul", defaults={"transpose_rhs": False})


# See torch.nn.functional.pad
# Args: input: AnyTensor, _pad: Sequence[int], mode: str = "constant",
#       value: Optional[float] = None
# Returns: AnyTensor
pad = create_overridable_op("pad", defaults={"mode": "constant", "value": None})


# Permute the tensor dimensions according to the permutation `dims` in line
# notation. The semantics are the same as torch.permute.
# Args: tensor: AnyTensor, dims: List[int]
# Returns: AnyTensor
permute = create_overridable_op("permute")


# See torch.mean
# Args: x: AnyTensor, dim: Union[int, List[int]], keepdim: bool = False,
#       dtype: torch.dtype = None
# Returns: AnyTensor
mean = create_overridable_op("mean", defaults={"keepdim": False, "dtype": None})


@overridable(is_trivially_replicable=False)
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
def quantize(
    tensor: AnyTensor, quantizer: AnyTensor, *, name: str = UnnamedTensorName
) -> AnyTensor:
    """Quantize a tensor using the provided quantizer."""
    ...


@quantize.trampoline
def _quantize_trampoline(
    d: SignatureDispatcher,
    tensor: AnyTensor,
    quantizer: AnyTensor,
    name: str = UnnamedTensorName,
) -> AnyTensor:
    dispatch_args = (tensor, quantizer)
    for override in d.find_overrides(dispatch_args):
        result = override(tensor, quantizer, name=name)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


# Reduces then splits/scatters across the devices.
# Args: tensor: AnyTensor, scatter_dim: int
# Returns: AnyTensor
reduce_scatter = create_overridable_op("reduce_scatter", is_trivially_replicable=False)


# Computes the full, unbiased RMS normalization of an input.
# Args: x: AnyTensor, weight: AnyTensor, epsilon: float, orig_dtype: torch.dtype
# Returns: AnyTensor
rms_norm = create_overridable_op("rms_norm")


# See torch.Tensor.repeat
# Args: input: AnyTensor, *sizes: List[int]
# Returns: AnyTensor
repeat = create_overridable_op("repeat")


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


# Computes the scaled dot product attention using QKV.
# Args: q: AnyTensor, k: AnyTensor, v: AnyTensor, a: Optional[AnyTensor],
#       is_causal: bool = False, scale: Optional[float] = None,
#       softcap: Optional[float] = None, impl: Optional[str] = None
# Returns: AnyTensor
scaled_dot_product_attention = create_overridable_op(
    "scaled_dot_product_attention",
    defaults={
        "a": None,
        "is_causal": False,
        "scale": None,
        "softcap": None,
        "impl": None,
    },
    dispatch_args=range(4),
)


# Returns a tensor with the same data and number of elements as input, but with
# the specified shape. See torch.reshape.
# Args: input: AnyTensor, shape: List[int]
# Returns: AnyTensor
reshape = create_overridable_op("reshape")


@overridable(is_trivially_replicable=False)
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


# Shard `input` the same way as `like`. This may require expensive resharding.
# Args: input: AnyTensor, like: AnyTensor
# Returns: AnyTensor
reshard_like = create_overridable_op("reshard_like", is_trivially_replicable=False)


@overridable
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


@scatter_.trampoline
def _scatter__trampoline(
    d: SignatureDispatcher,
    inout: AnyTensor,
    dim: int,
    index: AnyTensor,
    src: AnyTensor | Number,
    *,
    reduce: str = None,
) -> AnyTensor:
    dispatch_args = (inout, index, src)
    for override in d.find_overrides(dispatch_args):
        result = override(inout, dim, index, src, reduce=reduce)
        if result is not NotImplemented:
            return override, result
    else:
        d.fail(dispatch_args)


# See torch.scatter_add
# Args: input: AnyTensor, dim: int, index: AnyTensor, src: AnyTensor
# Returns: AnyTensor
scatter_add = create_overridable_op("scatter_add")


# Concats all shards along the sharding dimension.
# Does nothing if not sharded.
# Args: maybe_sharded: AnyTensor
# Returns: AnyTensor
sharded_cat = create_overridable_op("sharded_cat", is_trivially_replicable=False)


# Gather the input tensor from all devices to the given device ordinal.
# Args: input: AnyTensor, root_rank: int
# Returns: list[AnyTensor]
sharded_gather = create_overridable_op("sharded_gather", is_trivially_replicable=False)


# Return the shards of a sharded tensor.
# Args: input: ShardedTensor | QuantizedLayout
# Returns: list[AnyTensor | QuantizedLayout]
shards = create_overridable_op("shards", is_trivially_replicable=False)


# Reduce across the shards into a single device.
# root_rank:
#     Rank of receiving device within the tensor devices.
#     If sharded, `maybe_sharded.devices[root_rank]` is the destination.
# Args: maybe_sharded: AnyTensor, root_rank: int = 0
# Returns: AnyTensor
sharded_sum = create_overridable_op(
    "sharded_sum", is_trivially_replicable=False, defaults={"root_rank": 0}
)


# See torch.sigmoid
# Args: tensor: AnyTensor
# Returns: AnyTensor
sigmoid = create_overridable_op("sigmoid")


# See torch.nn.functional.softmax
# Args: tensor: AnyTensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None
# Returns: AnyTensor
softmax = create_overridable_op("softmax", defaults={"dim": None, "dtype": None})


# See torch.split
# Args: tensor: AnyTensor, split_size_or_sections: int | list[int], dim: int = 0
# Returns: tuple[AnyTensor, ...]
split = create_overridable_op("split", defaults={"dim": 0})


# SwiGLU activation function
# Args: tensor: AnyTensor, alpha: float = 1.702, limit: Optional[float] = None
# Returns: AnyTensor
swiglu = create_overridable_op("swiglu", defaults={"alpha": 1.702, "limit": None})


# See torch.Tensor.to
# Args: tensor: AnyTensor, *args, **kwargs
# Returns: AnyTensor
to = create_overridable_op("to")


# Trace tensor(s) in IREE runtime or in eager mode.
#
# You can add trace_tensor into your model wherever you want. It will insert a
# trace op into the IR. Then you can register a callback in the IREE runtime for
# custom handling of the trace command during execution. For example recording the
# tensor into a file. There is also a destination/sink for eager execution.
#
# The trace op will prevent fusion which will influence how the model is compiled.
# This may change the behavior of the program and cause a numerical issue to
# disappear if it was the result of op fusion.
#
# Example usage at sharktank/tests/ops/ops_test.py::TestTraceTensors.
#
# See:
# sharktank.utils.debugging.set_trace_tensor_callback
# sharktank.utils.debugging.trace_tensor_to_safetensors_callback
# sharktank.utils.debugging.flags.trace_path
# sharktank.utils.iree.make_hal_buffer_view_trace_default_callback
# sharktank.layers.BaseLayer.trace_tensor
#
# Args: key: str, *tensors: tuple[AnyTensor, ...]
trace_tensor = create_overridable_op("trace_tensor")


# Transfer the tensor to a device with ordinal `ordinal`.
# Args: tensor: AnyTensor, ordinal: int
# Returns: AnyTensor
barrier_on_logical_device = create_overridable_op(
    "barrier_on_logical_device", is_trivially_replicable=False
)


# Transfer the tensor to a device with ordinal `ordinal`.
# Args: tensor: AnyTensor, ordinal: int
# Returns: AnyTensor
transfer_to_logical_device = create_overridable_op(
    "transfer_to_logical_device", is_trivially_replicable=False
)


# See torch.transpose
# Args: tensor: AnyTensor, dim0: int, dim1: int
# Returns: AnyTensor
transpose = create_overridable_op("transpose")


# See torch.unflatten
# Args: input: AnyTensor, dim: int, sizes: Tuple[int]
# Returns: AnyTensor
unflatten = create_overridable_op("unflatten")


# Args: input: AnyTensor
# Returns: QuantizedLayout
unpack = create_overridable_op("unpack")


# Return the unpacked unscaled/quantized values of a block scales packed layout.
# Args: qs: AnyTensor, layout: BlockScaledPackedLayout
# Returns: AnyTensor
unpack_qs = create_overridable_op("unpack_qs")


# Return the tensor that has the same elements and shape, but is not sharded.
# Args: tensor: AnyTensor
# Returns: AnyTensor
unshard = create_overridable_op("unshard", is_trivially_replicable=False)


# See torch.unsqueeze
# Args: tensor: AnyTensor, dim: int
# Returns: AnyTensor
unsqueeze = create_overridable_op("unsqueeze")


# See torch.squeeze
# Args: tensor: AnyTensor, dim: Optional[int]
# Returns: AnyTensor
squeeze = create_overridable_op("squeeze")


# See torch.sum
# Args: input: AnyTensor, dim: Union[int, List[int]], keepdim: bool = False,
#       dtype: torch.dtype = None
# Returns: AnyTensor
sum = create_overridable_op("sum", defaults={"keepdim": False, "dtype": None})


# TODO: Convert this to use the factory function
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


# See torch.Tensor.view
# Args: tensor: AnyTensor, shape: Optional[List[int]] = None,
#       dtype: Optional[torch.dtype] = None
# Returns: AnyTensor
view = create_overridable_op("view", defaults={"shape": None, "dtype": None})


# See torch.Tensor.view_as_complex
# Args: tensor: AnyTensor
# Returns: AnyTensor
view_as_complex = create_overridable_op("view_as_complex")


# See torch.Tensor.view_as_real
# Args: tensor: AnyTensor
# Returns: AnyTensor
view_as_real = create_overridable_op("view_as_real")


# See torch.zeros_like
# Args: tensor: AnyTensor, dtype: Optional[torch.dtype] = None,
#       layout: Optional[torch.layout] = None, device: Optional[torch.device] = None,
#       requires_grad: bool = False, memory_format: torch.memory_format = torch.preserve_format
# Returns: AnyTensor
zeros_like = create_overridable_op(
    "zeros_like",
    defaults={
        "dtype": None,
        "layout": None,
        "device": None,
        "requires_grad": False,
        "memory_format": torch.preserve_format,
    },
)
