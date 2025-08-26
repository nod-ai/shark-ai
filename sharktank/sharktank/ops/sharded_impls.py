# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import List, Optional, Sequence, Union, Tuple
import itertools
from numbers import Number
import math
import functools
import torch
from torch import Tensor

from sharktank.types import (
    AnyTensor,
    BlockScaledFp4Layout,
    BlockScaledLayout,
    DefaultPrimitiveTensor,
    InferenceTensor,
    is_any_tensor,
    PrimitiveTensor,
    QuantizedLayout,
    ReplicatedTensor,
    ShardedTensor,
    sharding,
    SplitPrimitiveTensor,
    StaticScaledQuantizer,
    Theta,
    UnnamedTensorName,
    UnreducedTensor,
)
from sharktank.types.tensors import unbox_tensor, is_any_tensor
from ._registry import (
    AllOfExprs,
    AllOfType,
    AllOfExprsVariadic,
    AnyOfType,
    BoolTypeExpr,
    IsOfType,
    SignatureDispatcher,
    get_all_registered_ops,
)
from .shape import (
    broadcast_dims,
    broadcast_dim,
    unbroadcast_dim,
    normalize_negative_dim,
)
from sharktank.utils import longest_equal_range, tree
from sharktank.utils.math import ceildiv
from .signatures import *
from .shaping import expand, flatten, permute, transpose, unflatten, unsqueeze
from .sharding.utils import assert_on_same_devices, transfer_n_pin, wrap_override


def sharded_wrap_override():
    do_not_wrap = {
        "all_gather",
        "all_reduce",
        "equal",
        "index_copy_",
        "index_put_",
        "replicate_like",
        "replicate",
        "reshard_like",
        "trace_tensor",
        "transfer_to_logical_device",
        "unshard",
    }

    for func_name, func in get_all_registered_ops().items():
        if func_name not in do_not_wrap and hasattr(func, "override"):
            func.override_orig = func.override
            func.override = wrap_override(func.override_orig)


def sharded_unwrap_override():
    """
    Unwraps [op].override to restore the original function.
    Must be called at the end of this file.
    """
    from . import signatures

    for func_name in signatures.__all__:
        func = globals()[func_name]
        if hasattr(func, "override_orig"):
            func.override = func.override_orig
            del func.override_orig


def _register_trivially_replicable():
    from .utils import trivially_replicable

    def replicated_if_tensor(t: type) -> bool:
        if issubclass(t, ReplicatedTensor):
            return True
        if not issubclass(t, (torch.Tensor, InferenceTensor)):
            return True
        return False

    def should_override(*types: tuple[type]) -> bool:
        at_least_one_replicated_tensor = any(
            issubclass(t, ReplicatedTensor) for t in types
        )
        if not at_least_one_replicated_tensor:
            return False
        return all(replicated_if_tensor(t) for t in types)

    for func_name, func in get_all_registered_ops().items():
        if isinstance(func, SignatureDispatcher) and func.is_trivially_replicable:
            func.override(BoolTypeExpr(should_override))(trivially_replicable(func))


sharded_wrap_override()

_register_trivially_replicable()


@all_gather.override(SplitPrimitiveTensor)
def all_gather_split(
    input: SplitPrimitiveTensor, *, dim: int | None
) -> ReplicatedTensor:
    dim = input.shard_dim if dim is None else dim

    gathered = cat(
        [
            (
                transfer_to_logical_device(shard, input.devices[0])
                if i != 0
                else barrier_on_logical_device(shard, input.devices[0])
            )
            for i, shard in enumerate(input.shards)
        ],
        dim=dim,
    )
    shards = [
        (
            transfer_to_logical_device(gathered, input.devices[i])
            if i != 0
            else barrier_on_logical_device(gathered, input.devices[0])
        )
        for i in range(input.shard_count)
    ]
    return ReplicatedTensor(ts=shards, devices=input.devices)


@all_reduce.override(AllOfType(SplitPrimitiveTensor, UnreducedTensor))
def all_reduce_split_or_unreduced(
    input: Union[SplitPrimitiveTensor, UnreducedTensor],
) -> ReplicatedTensor:
    if len(input.shards) == 1:
        return ReplicatedTensor(ts=input.shards, devices=input.devices)

    reduced = functools.reduce(
        lambda x, y: elementwise(torch.add, x, y),
        [
            (
                transfer_to_logical_device(shard, input.devices[0])
                if i != 0
                else barrier_on_logical_device(shard, input.devices[0])
            )
            for i, shard in enumerate(input.shards)
        ],
    )
    shards = [
        (
            transfer_to_logical_device(reduced, input.devices[i])
            if i != 0
            else barrier_on_logical_device(reduced, input.devices[0])
        )
        for i in range(input.shard_count)
    ]
    return ReplicatedTensor(ts=shards, devices=input.devices)


@argmax.override(ReplicatedTensor)
def argmax_replicated(
    tensor: ReplicatedTensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
):
    shards = [argmax(shard, dim, keepdim, chunk_size) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@argmax.override(SplitPrimitiveTensor)
def argmax_split(
    tensor: SplitPrimitiveTensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    chunk_size: Optional[int] = None,
):
    shards = [argmax(shard, dim, keepdim, chunk_size) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


def attention_mask_replicated(
    boolean_input_mask: ReplicatedTensor,
    start_positions: ReplicatedTensor | None,
    *,
    attention_dtype: torch.dtype,
) -> ReplicatedTensor:
    start_pos_shards = [None] * len(boolean_input_mask.shards)
    if start_positions is not None:
        start_pos_shards = start_positions.shards

    shards = [
        attention_mask(bool_mask, start_pos, attention_dtype=attention_dtype)
        for bool_mask, start_pos in zip(boolean_input_mask.shards, start_pos_shards)
    ]

    return ReplicatedTensor(ts=shards, devices=boolean_input_mask.devices)


attention_mask.override(ReplicatedTensor, ReplicatedTensor)(attention_mask_replicated)
attention_mask.override(ReplicatedTensor)(attention_mask_replicated)


@cat.override(AllOfType(SplitPrimitiveTensor))
def cat_split(
    tensors: Sequence[SplitPrimitiveTensor], dim: int
) -> SplitPrimitiveTensor:
    assert len(tensors) > 0
    assert all(
        [
            t.shard_count == tensors[0].shard_count
            and t.shard_dim == tensors[0].shard_dim
            for t in tensors
        ]
    )
    shard_dim = tensors[0].shard_dim
    shard_count = tensors[0].shard_count
    if dim != shard_dim:
        shards = [cat(shards, dim) for shards in zip(*[t.shards for t in tensors])]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)
    else:
        # TODO: implement efficient cat along split dim.
        # This would probably result in doing the concatenation on one device.
        concatenated_unsharded = cat(
            [shard for t in tensors for shard in t.shards], dim
        )
        return reshard_split(
            concatenated_unsharded,
            dim=shard_dim,
            count=shard_count,
            devices=tensors[0].devices,
        )


# conv2d


def conv2d_all_split(
    input: SplitPrimitiveTensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        input.is_replicated or input.shard_dim == 1
    ), "Only sharding of input channel dimension is supported"
    assert (
        bias is None or weight.shard_dim == 0 and bias.shard_dim == 0
    ), "Only sharding of output channel dimension is supported"

    # TODO: allow for implementation where we don't all-gather, but gather
    # instead and share the input tensor.
    # This may be useful when having peered memory.
    #
    # Another option is to have each device do multiple convolutions without
    # doing an gather/all-gather.
    # Then a reduction across the shards.
    # If groups are divisible by the number of shards we don't need to do a
    # reduction.
    # We would be relaying on the compiler to fuse the convs into a single
    # kernel.
    # A batched conv where the mini-batches(shards) are scattered across
    # multiple buffers.
    #
    # With tuning allow for selection of the appropriate version.

    input = all_gather(input)

    return conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


conv2d.override(
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    auto_dequant=True,
)(conv2d_all_split)
conv2d.override(SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_all_split
)


def conv2d_replicated_input_split_weight_and_bias(
    input: ReplicatedTensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    assert input.shard_count == weight.shard_count
    assert bias is None or weight.shard_count == bias.shard_count
    assert (
        bias is None or weight.shard_dim == 0 and bias.shard_dim == 0
    ), "Only sharding of output channel dimension is supported"
    assert groups == 1

    shards = [
        conv2d(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        for x, w, b in zip(
            input.shards,
            weight.shards,
            [None] * weight.shard_count if bias is None else bias.shards,
        )
    ]
    return SplitPrimitiveTensor(shard_dim=1, ts=shards)


conv2d.override(
    ReplicatedTensor, SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True
)(conv2d_replicated_input_split_weight_and_bias)
conv2d.override(ReplicatedTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_replicated_input_split_weight_and_bias
)


def conv2d_split_weight_and_bias(
    input: Tensor,
    weight: SplitPrimitiveTensor,
    bias: SplitPrimitiveTensor | None,
    *,
    stride,
    padding,
    dilation,
    groups,
    accum_dtype,
) -> SplitPrimitiveTensor:
    assert accum_dtype is None, "accum_dtype not supported"
    if bias is not None:
        assert weight.shard_count == bias.shard_count

    # Output channels dimension is split.
    if weight.shard_dim == 0 and groups == 1:
        assert bias is None or bias.shard_dim == 0
        shards = [
            conv2d(
                input,
                w,
                b,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            for w, b in zip(
                weight.shards,
                [None] * weight.shard_count if bias is None else bias.shards,
            )
        ]
        return SplitPrimitiveTensor(shard_dim=1, ts=shards)
    else:
        assert False, "Unsupported, TODO: handle split channels in input"


conv2d.override(Tensor, SplitPrimitiveTensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_split_weight_and_bias
)
conv2d.override(Tensor, SplitPrimitiveTensor, auto_dequant=True)(
    conv2d_split_weight_and_bias
)


@dequantize.override(dict, ReplicatedTensor)
def dequantize_planes_split_replicated_static_scaled_quantizer(
    input: dict[str, SplitPrimitiveTensor],
    quantizer: ReplicatedTensor,
    dtype: torch.dtype | None,
) -> SplitPrimitiveTensor:
    qs = input["qs"]
    if not isinstance(qs, SplitPrimitiveTensor) or not isinstance(
        quantizer.shards[0], StaticScaledQuantizer
    ):
        return NotImplemented

    shards = [
        dequantize({"qs": qs_shard}, quantizer=quantizer_shard, dtype=dtype)
        for qs_shard, quantizer_shard in zip(qs.shards, quantizer.shards, strict=True)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=qs.shard_dim, devices=qs.devices)


# Sharded elementwise.


@elementwise.override(SplitPrimitiveTensor)
def split_elementwise_unary(operator, x: SplitPrimitiveTensor, *args, **kwargs):
    partials = [operator(unbox_tensor(pt), *args, **kwargs) for pt in x.shards]
    return SplitPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def split_elementwise_binary(
    operator, x: SplitPrimitiveTensor, y: SplitPrimitiveTensor, *args, **kwargs
):
    assert x.shard_count == y.shard_count
    x_shard_dim, y_shard_dim = broadcast_dims([x.shard_dim, y.shard_dim], [x, y])
    assert x_shard_dim == y_shard_dim
    pt_xs = [unbox_tensor(pt) for pt in x.shards]
    pt_ys = [unbox_tensor(pt) for pt in y.shards]
    partials = [
        operator(pt_x, pt_y, *args, **kwargs) for pt_x, pt_y in zip(pt_xs, pt_ys)
    ]
    return SplitPrimitiveTensor(
        shard_dim=x_shard_dim,
        shape=torch.broadcast_shapes(x.shape, y.shape),
        ts=partials,
    )


@elementwise.override(SplitPrimitiveTensor, Number)
def elementwise_binary_split_lhs_scalar_rhs(
    operator, x: SplitPrimitiveTensor, y: Number, *args, **kwargs
):
    pt_xs = [unbox_tensor(pt) for pt in x.shards]
    partials = [operator(pt_x, y, *args, **kwargs) for pt_x in pt_xs]
    return SplitPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(SplitPrimitiveTensor, Tensor)
def elementwise_binary_split_lhs_tensor_rhs(
    operator, x: SplitPrimitiveTensor, y: Tensor, *args, **kwargs
):
    return elementwise(operator, x, reshard_like(y, like=x), *args, **kwargs)


@elementwise.override(ReplicatedTensor, SplitPrimitiveTensor)
def elementwise_binary_replicated_lhs_sharder_rhs(
    operator, x: ReplicatedTensor, y: SplitPrimitiveTensor, *args, **kwargs
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    # A replicated tensor can be split with no cost.
    # It is natural to propagate the split instead of the replication.
    x_sharded = reshard_like(x, like=y)
    return elementwise(operator, x_sharded, y, *args, **kwargs)


@elementwise.override(SplitPrimitiveTensor, ReplicatedTensor)
def elementwise_binary_split_lhs_replicated_rhs(
    operator, x: SplitPrimitiveTensor, y: ReplicatedTensor, *args, **kwargs
):
    assert len(y.shape) > 0, "0-rank not supported"
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )

    shard_dim_in_res = broadcast_dim(x.shard_dim, [x.shape, y.shape])
    shard_dim_in_y = unbroadcast_dim(shard_dim_in_res, [y.shape, x.shape])
    is_shard_dim_broadcasted_in_y = (
        shard_dim_in_y is None or y.shape[shard_dim_in_y] == 1
    )
    if is_shard_dim_broadcasted_in_y:
        shards = [
            elementwise(operator, x_shard, y_shard)
            for x_shard, y_shard in zip(x.shards, y.shards)
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim_in_res)

    y_sharded = reshard_like(y, like=x)
    return elementwise(operator, x, y_sharded, *args, **kwargs)


@elementwise.override(ReplicatedTensor, UnreducedTensor)
def elementwise_binary_replicated_lhs_unreduced_rhs(
    operator, x: ReplicatedTensor, y: UnreducedTensor, *args, **kwargs
):
    if x.shard_count != y.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({x.shard_count} != {y.shard_count})"
        )
    y_replicated = reshard_like(y, like=x)
    return elementwise(operator, x, y_replicated, *args, **kwargs)


@elementwise.override(ReplicatedTensor, Tensor)
def elementwise_binary_replicated_lhs_unsharded_rhs(
    operator, x: ReplicatedTensor, y: Tensor, *args, **kwargs
):
    y_replicated = reshard_like(y, like=x)
    return elementwise(operator, x, y_replicated, *args, **kwargs)


@elementwise.override(Tensor, ReplicatedTensor)
def elementwise_binary_replicated_lhs_unsharded_rhs(
    operator, x: Tensor, y: ReplicatedTensor, *args, **kwargs
):
    x_replicated = reshard_like(x, like=y)
    return elementwise(operator, x_replicated, y, *args, **kwargs)


@expand.override(SplitPrimitiveTensor)
def expand_split(
    tensor: SplitPrimitiveTensor, shape: List[int]
) -> SplitPrimitiveTensor:
    assert len(shape) == len(tensor.shape)
    shard_dim = tensor.shard_dim
    not_expanding_split_dim = (
        shape[shard_dim] == -1 or shape[shard_dim] == tensor.shape[shard_dim]
    )
    assert not_expanding_split_dim, "Expanding a split dimension is not supported"

    shape = list(shape)
    shape[shard_dim] = -1
    shards = [expand(shard, shape) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@group_norm_affine.override(
    SplitPrimitiveTensor, SplitPrimitiveTensor, SplitPrimitiveTensor
)
def shareded_group_norm_affine(input, weight, bias, *, num_groups, eps):
    assert (
        input.shard_count == weight.shard_count
        and input.shard_count == bias.shard_count
    )
    assert input.shard_dim == 1, "Can shard only the channel dimension"
    assert num_groups % input.shard_count == 0, "Can shard only groups"
    num_groups_per_shard = num_groups // input.shard_count

    result_shards = [
        group_norm_affine(x, num_groups=num_groups_per_shard, weight=w, bias=b, eps=eps)
        for x, w, b in zip(input.shards, weight.shards, bias.shards)
    ]

    return SplitPrimitiveTensor(shard_dim=1, ts=result_shards)


@index_copy_.override(SplitPrimitiveTensor, ReplicatedTensor, ReplicatedTensor)
def index_copy__split_replicated_replicated(
    inout: SplitPrimitiveTensor,
    dim: int,
    index: ReplicatedTensor,
    tensor: ReplicatedTensor,
) -> SplitPrimitiveTensor:
    assert (
        inout.shard_count == index.shard_count
        and inout.shard_count == tensor.shard_count
    )
    assert inout.shard_dim != dim
    for inout_shard, index_shard, tensor_shard in zip(
        inout.shards, index.shards, tensor.shards
    ):
        index_copy_(inout_shard, dim, index_shard, tensor_shard)
    return inout


@index_copy_.override(SplitPrimitiveTensor, ReplicatedTensor, SplitPrimitiveTensor)
def index_copy__split_replicated_split(
    inout: SplitPrimitiveTensor,
    dim: int,
    index: ReplicatedTensor,
    tensor: SplitPrimitiveTensor,
) -> SplitPrimitiveTensor:
    assert (
        inout.shard_count == index.shard_count
        and inout.shard_count == tensor.shard_count
    )
    assert inout.shard_dim == tensor.shard_dim
    assert inout.shard_dim != dim
    for inout_shard, index_shard, tensor_shard in zip(
        inout.shards, index.shards, tensor.shards
    ):
        index_copy_(inout_shard, dim, index_shard, tensor_shard)
    return inout


@index_put_.override(
    AllOfExprsVariadic(
        IsOfType(SplitPrimitiveTensor),
        IsOfType(SplitPrimitiveTensor),
        IsOfType(Tensor, PrimitiveTensor, ReplicatedTensor),
    )
)
def index_put__split(
    inout: SplitPrimitiveTensor,
    indices: Tuple[Union[Tensor, PrimitiveTensor, ReplicatedTensor]],
    values: SplitPrimitiveTensor,
) -> SplitPrimitiveTensor:
    # TODO: verify that the values split dimension is not being indexed or implement
    # this case.
    indices = [replicate(idx, count=inout.shard_count) for idx in indices]
    for i, shard in enumerate(inout.shards):
        shard_indices = [idx.shards[i] for idx in indices]
        shard.index_put_(shard_indices, values.shards[i])
    return inout


@index_select.override(SplitPrimitiveTensor, ReplicatedTensor)
def index_select_split_replicated(
    tensor: SplitPrimitiveTensor,
    dim: int,
    index: ReplicatedTensor,
) -> ReplicatedTensor:
    assert tensor.shard_count == index.shard_count
    assert (
        dim != tensor.shard_dim
    ), "Indexing along the split dimension is not supported."
    shards = [
        index_select(tensor_shard, dim, index_shard)
        for tensor_shard, index_shard in zip(tensor.shards, index.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@interpolate.override(SplitPrimitiveTensor)
def interpolate_split_batch_or_channel(
    input: SplitPrimitiveTensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> SplitPrimitiveTensor:
    assert input.shard_dim == 0 or input.shard_dim == 1
    shards = [
        torch.nn.functional.interpolate(
            input=unbox_tensor(shard),
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        for shard in input.shards
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=input.shard_dim)


@layer_norm.override(SplitPrimitiveTensor, Tensor, Tensor)
def layer_norm_split(
    input, weight, bias, *, eps, normalized_shape: Optional[tuple[int]]
):
    assert input.shard_dim >= 0 and input.shard_dim < len(input.shape) - len(
        weight.shape
    )
    shards = [
        layer_norm(shard, weight, bias, eps=eps, normalized_shape=normalized_shape)
        for shard in input.shards
    ]
    return SplitPrimitiveTensor(shard_dim=input.shard_dim, ts=shards)


# Linear
def linear_sharded(
    input: Tensor | ShardedTensor,
    weight: Tensor | ShardedTensor,
    bias: Tensor | ShardedTensor | None,
    *,
    accum_dtype,
    matmul_impl=None,
) -> SplitPrimitiveTensor:
    # TODO: handle different dtypes
    result = matmul(input, weight, transpose_rhs=True, impl=matmul_impl)
    if bias is not None:
        result = elementwise(torch.add, result, bias)
    return result


# Override for all cases of Tensor or ShardedTensor arguments,
# except when all Tensors.
# Then we want the default implementation to handle it.
for types in itertools.product([Tensor, ShardedTensor], repeat=3):
    if tuple(types) != (Tensor,) * 3:
        linear.override(*types, auto_dequant=True)(linear_sharded)
for types in itertools.product([Tensor, ShardedTensor], repeat=2):
    if tuple(types) != (Tensor,) * 2:
        linear.override(*types, auto_dequant=True)(linear_sharded)


@masked_fill.override(AllOfType(SplitPrimitiveTensor))
def masked_fill_split(
    tensor: SplitPrimitiveTensor,
    mask: SplitPrimitiveTensor,
    value: Number,
) -> SplitPrimitiveTensor:
    assert tensor.shard_count == mask.shard_count
    shards = [
        shard.masked_fill(mask_shard, value)
        for shard, mask_shard in zip(tensor.shards, mask.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


# Sharded matmuls.


@matmul.override(ReplicatedTensor, SplitPrimitiveTensor)
def matmul_replicated_lhs_split_rhs(
    lhs: ReplicatedTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor | UnreducedTensor:
    assert lhs.shard_count == rhs.shard_count

    if transpose_rhs:
        assert len(rhs.shape) > 1
        rhs_reduction_dim = len(rhs.shape) - 1
    else:
        rhs_reduction_dim = len(rhs.shape) - 2 if len(rhs.shape) > 1 else 0
    if rhs_reduction_dim == rhs.shard_dim:
        lhs_reduction_dimension = len(lhs.shape) - 1
        lhs_split = reshard_split(
            lhs, dim=lhs_reduction_dimension, count=lhs.shard_count
        )
        return matmul(lhs_split, rhs, transpose_rhs=transpose_rhs)

    is_batched_rhs = len(rhs.shape) > 2
    is_rhs_batch_dim_split = is_batched_rhs and rhs.shard_dim < len(rhs.shape) - 2
    if is_rhs_batch_dim_split:
        assert len(lhs.shape) == len(rhs.shape), "TODO: implement general case"
        lhs_split = reshard_split(lhs, dim=rhs.shard_dim, count=lhs.shard_count)
        return matmul(lhs_split, rhs, transpose_rhs=transpose_rhs)

    # The RHS parallel dimension is split.
    shards = [
        matmul(lhs_shard, rhs_shard, transpose_rhs=transpose_rhs)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=len(shards[0].shape) - 1)


@matmul.override(SplitPrimitiveTensor, Tensor)
def matmul_split_lhs(
    lhs: SplitPrimitiveTensor, rhs, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert lhs_reduction_dim != lhs.shard_dim
    shards = [
        matmul(lhs_shard, rhs, transpose_rhs=transpose_rhs) for lhs_shard in lhs.shards
    ]
    return SplitPrimitiveTensor(shard_dim=lhs.shard_dim, ts=shards)


@matmul.override(Tensor, SplitPrimitiveTensor)
def matmul_split_rhs(
    lhs, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    # When multiplying (unsharded, split), the rhs must be split by column.
    # In a transposed configuration, this is axis 0, otherwise 1.
    # This will result in a ShardedTensor, split by column.
    lhs = unbox_tensor(lhs)
    rhs_shard_dim = rhs.shard_dim
    if transpose_rhs:
        assert (
            rhs_shard_dim == 0
        ), f"matmul[split, transposed rhs] must be split on dim 0 but is {rhs_shard_dim}"
    else:
        assert (
            rhs_shard_dim == 1
        ), f"matmul[split rhs] must be split on dim 1 but is {rhs_shard_dim}"
    partials = [
        matmul(lhs, partial_rhs, transpose_rhs=transpose_rhs)
        for partial_rhs in rhs.shards
    ]
    # The partial is split columnwise (last dim).
    return SplitPrimitiveTensor(shard_dim=len(lhs.shape) - 1, ts=partials)


@matmul.override(SplitPrimitiveTensor, ReplicatedTensor)
def matmul_split_lhs_replicated_rhs(
    lhs: SplitPrimitiveTensor, rhs: ReplicatedTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor:
    lhs_reduction_dim = len(lhs.shape) - 1
    assert (
        lhs_reduction_dim != lhs.shard_dim
    ), "TODO: implement split reduction dimension"
    is_lhs_batched = len(lhs.shape) > 2
    is_rhs_batched = len(rhs.shape) > 2
    is_lhs_batch_dim_split = lhs.shard_dim < len(lhs.shape) - 2
    if is_lhs_batch_dim_split:
        assert not (
            is_rhs_batched and is_lhs_batched
        ), "TODO: implement when LHS has a split batch dim and RHS has a batch dim"
    shards = [
        matmul(lhs_shard, rhs_shard, transpose_rhs=transpose_rhs)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    shard_dim = lhs.shard_dim + max(0, len(rhs.shape) - len(lhs.shape))
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@matmul.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def matmul_split(
    lhs: SplitPrimitiveTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> UnreducedTensor | SplitPrimitiveTensor:
    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul split tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )

    lhs_reduction_dim = len(lhs.shape) - 1
    if transpose_rhs:
        assert len(rhs.shape) > 1, "Vector rhs not supported"
        rhs_reduction_dim = len(rhs.shape) - 1
    else:
        rhs_reduction_dim = len(rhs.shape) - 2 if len(rhs.shape) > 1 else 0

    # The reduction dimension is split on both tensors.
    if lhs_reduction_dim == lhs.shard_dim and rhs_reduction_dim == rhs.shard_dim:
        partials = [
            matmul(partial_lhs, partial_rhs, transpose_rhs=transpose_rhs)
            for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
        ]
        return UnreducedTensor(ts=partials)

    is_batched_matmul = len(lhs.shape) > 2 or len(rhs.shape) > 2
    if (
        is_batched_matmul
        and len(lhs.shape) == len(rhs.shape)
        and lhs.shard_dim == rhs.shard_dim
        and lhs.shard_dim < len(lhs.shape) - 2
    ):
        # The same batch dim is sharded for both arguments.
        shards = [
            matmul(lhs_shard, rhs_shard, transpose_rhs=transpose_rhs)
            for lhs_shard, rhs_shard in zip(lhs.shards, rhs.shards)
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)

    # -1 for missing parallel dim.
    lhs_parallel_dim = len(lhs.shape) - 2
    if transpose_rhs:
        rhs_parallel_dim = len(rhs.shape) - 2 if len(rhs.shape) > 1 else -1
    else:
        rhs_parallel_dim = len(rhs.shape) - 1 if len(rhs.shape) > 1 else -1

    # One parallel dimension is split for each tensor.
    # Or lhs batch dim and rhs parallel dim are split.
    if lhs.shard_dim <= lhs_parallel_dim and rhs_parallel_dim == rhs.shard_dim:
        # We gather along the rhs shard dim.
        # It is more natural to preserve the sharding axis of the input.
        # TODO: This assumes non-peered memory. We prepare the operands to be
        # available on the required devices.
        # We need to distinguish based on some config.
        replicated_rhs = replicate(rhs, count=lhs.shard_count)
        return matmul(lhs, replicated_rhs, transpose_rhs=transpose_rhs)

    assert False, "Sharding configuration not supported"


@scaled_dot_product_attention.override(
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    Optional[ReplicatedTensor],
    impl_name="sharded",
)
def scaled_dot_product_attention_sharded(
    q, k, v, a, sink, sliding_window, is_causal, scale, softcap, impl
) -> SplitPrimitiveTensor:
    if sink is not None or sliding_window is not None:
        return NotImplemented
    if q.shard_count != k.shard_count or q.shard_count != v.shard_count:
        raise ValueError("Incompatible number of shards for qkv")

    if a and q.shard_count != a.shard_count:
        raise ValueError(
            f"Incompatible number of shards for a ({a.shard_count}) should be ({q.shard_count})"
        )

    if q.shard_dim != k.shard_dim or q.shard_dim != v.shard_dim:
        raise ValueError("Incompatible shard dim across qkv")

    if q.shard_dim > len(q.shards[0].shape) - 2:
        raise ValueError("Sharding must occur as batch dimension")

    a_shards = [None] * q.shard_count
    if a is not None:
        a_shards = a.shards

    output_shards = []
    for q_s, k_s, v_s, a_s in zip(q.shards, k.shards, v.shards, a_shards):
        o_s = scaled_dot_product_attention(
            q_s,
            k_s,
            v_s,
            a_s,
            is_causal=is_causal,
            scale=scale,
            softcap=softcap,
            impl=impl,
        )
        output_shards.append(o_s)

    return SplitPrimitiveTensor(ts=output_shards, shard_dim=q.shard_dim)


@mean.override(SplitPrimitiveTensor)
def mean_split(
    x: SplitPrimitiveTensor,
    dim: Union[int, List[int]],
    keepdim: bool,
    *,
    dtype: torch.dtype,
) -> SplitPrimitiveTensor | ReplicatedTensor:
    if not isinstance(dim, (list, tuple)):
        dim = [dim]
    dim = [d + len(x.shape) if d < 0 else d for d in dim]

    if x.shard_dim not in dim:
        # If keepdim == False and any entry in dim is smaller than shard_dim
        # we need to offset shard_dim_new to have it point to the same dimension.
        num_smaller_dims = sum(d < x.shard_dim for d in dim)
        shard_dim_new = x.shard_dim - (not keepdim) * num_smaller_dims

        shards = [
            mean(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in x.shards
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim_new)
    else:

        partial_sums = [
            sum(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in x.shards
        ]
        # reduce to x.devices[0] for now - TODO: use all_reduce once IREE supports it
        total_sum = sharded_sum(UnreducedTensor(ts=partial_sums, devices=x.devices))

        total_cnt = math.prod(x.shape[d] for d in dim)

        global_mean = total_sum / total_cnt

        return ReplicatedTensor(
            ts=global_mean, shard_count=x.shard_count, devices=x.devices
        )


@module_register_buffer.override(torch.nn.Module, ShardedTensor)
def module_register_buffer_sharded(
    module: torch.nn.Module, name: str, tensor: ShardedTensor
) -> None:
    for i, shard in enumerate(tensor.shards):
        module_register_buffer(module, f"{name}__shard__{i}", shard)
    setattr(module, name, tensor)


@pad.override(SplitPrimitiveTensor)
def pad_split(
    input: SplitPrimitiveTensor,
    _pad: List[int],
    mode: str = None,
    value: Optional[float] = None,
) -> SplitPrimitiveTensor:
    assert len(_pad) % 2 == 0, "Pad must be a list of even length"
    padding_shard_dim = input.shard_dim > (len(input.shape) - 1 - len(_pad) // 2)
    if padding_shard_dim:
        # If padding by 0, then it's not really padding and we can avoid transfers.
        shard_dim_indx_from_back = (len(input.shape) - 1) - input.shard_dim
        shard_dim_pads = _pad[shard_dim_indx_from_back : shard_dim_indx_from_back + 2]
        padding_shard_dim &= any(pad > 0 for pad in shard_dim_pads)
    if not padding_shard_dim:
        shards = [
            pad(shard, _pad=_pad, mode=mode, value=value) for shard in input.shards
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=input.shard_dim)
    else:
        gathered = cat(
            [
                (
                    transfer_to_logical_device(shard, input.devices[0])
                    if i != 0
                    else barrier_on_logical_device(shard, input.devices[0])
                )
                for i, shard in enumerate(input.shards)
            ],
            dim=input.shard_dim,
        )
        gathered = pad(gathered, _pad=_pad, mode=mode, value=value)
        return reshard_split(
            gathered,
            dim=input.shard_dim,
            count=input.shard_count,
            devices=input.devices,
        )


@quantize.override(SplitPrimitiveTensor, ShardedTensor)
def quantize_split(
    tensor: SplitPrimitiveTensor, quantizer: ShardedTensor, name: str
) -> SplitPrimitiveTensor:
    shards = [
        quantize(tensor_shard, quantizer_shard)
        for tensor_shard, quantizer_shard in zip(tensor.shards, quantizer.shards)
    ]
    return tensor.clone(ts=shards, name=name)


@reduce_scatter.override(UnreducedTensor)
def reduce_scatter(tensor: UnreducedTensor, scatter_dim: int) -> SplitPrimitiveTensor:
    # The performance here is contingent on the ability to have multiple transfers in
    # flight between devices.
    # Another approach is to reduce into a single device and then scatter.
    # The approach here moves strictly less data between devices but it would have
    # higher overhead due to having more transfer ops. What is better would depend
    # on the size of the tensor. For a 2-device case this should be better.

    if scatter_dim < 0:
        scatter_dim = len(tensor.shape) + scatter_dim
    assert scatter_dim < len(tensor.shape)

    unreduced_pieces: tuple[UnreducedTensor, ...] = split(
        tensor, ceildiv(tensor.shape[scatter_dim], tensor.shard_count), dim=scatter_dim
    )
    reduced_shards = [
        sharded_sum(t, root_rank=i) for i, t in enumerate(unreduced_pieces)
    ]
    return SplitPrimitiveTensor(ts=reduced_shards, shard_dim=scatter_dim)


@replicate.override(ReplicatedTensor)
def replicate_replicated(
    input: ReplicatedTensor, *, count: int, devices: None
) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    return input


@replicate.override(SplitPrimitiveTensor)
def replicate_split(
    input: SplitPrimitiveTensor, *, count: int, devices: None
) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    return all_gather(input)


@replicate.override(UnreducedTensor)
def replicate_unreduced(
    input: UnreducedTensor, *, count: int, devices: None
) -> ReplicatedTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    return all_reduce(input)


@replicate.override(Tensor)
def replicate_unsharded(input, *, count: int, devices: Tuple[int]) -> ReplicatedTensor:
    torch_input = unbox_tensor(input)
    assert count == len(devices)
    return ReplicatedTensor(ts=torch_input, shard_count=count, devices=devices)


@reshard.override(
    AllOfExprs(IsOfType(Tensor, InferenceTensor), IsOfType(sharding.Split))
)
def reshard_tensor_split(input: AnyTensor, spec: sharding.Split) -> AnyTensor:
    return reshard_split(input, dim=spec.shard_dim, count=spec.shard_count)


@reshard.override(Theta, sharding.ThetaLayerSharding)
def reshard_theta_layer_sharding(
    input: Theta, spec: sharding.ThetaLayerSharding
) -> Theta:
    return reshard(input, spec.theta_sharding())


@reshard.override(Theta, sharding.ThetaSharding)
def reshard_theta_sharding(input: Theta, spec: sharding.ThetaSharding) -> Theta:
    def make_value(input: Theta | InferenceTensor, spec) -> dict | InferenceTensor:
        result = reshard(input, spec)
        if isinstance(result, Theta):
            result = result.tree
        elif isinstance(result, torch.Tensor):
            result = DefaultPrimitiveTensor(data=result, name=input.name)
        else:
            assert isinstance(result, InferenceTensor)
            result.name = input.name
        return result

    return Theta(
        {
            k: make_value(input(k), spec[k])
            for k in input.keys
            if not isinstance(spec[k], sharding.Ignore)
        }
    )


@reshard.override(Theta, sharding.ThetaLayerSharding)
def reshard_theta_layer_sharding(
    input: Theta, spec: sharding.ThetaLayerSharding
) -> Theta:
    return reshard(input, spec.theta_sharding())


@reshard.override(object, sharding.Unsharded)
def reshard_all_to_unsharded(input: AnyTensor, spec: sharding.Unsharded) -> Tensor:
    return unshard(input)


@reshard.override(object, sharding.Replicated)
def reshard_all_to_replicated(
    input: AnyTensor, spec: sharding.Replicated
) -> ReplicatedTensor:
    return replicate(input, spec.shard_count)


@reshard_split.override(IsOfType(Tensor, InferenceTensor))
def reshard_split_unsharded(
    input: AnyTensor, *, dim: int, count: int, devices: tuple[int, ...]
) -> SplitPrimitiveTensor:
    dim = normalize_negative_dim(input, dim)
    return SplitPrimitiveTensor(
        ts=input, shard_dim=dim, shard_count=count, devices=devices
    )


@reshard_split.override(SplitPrimitiveTensor)
def reshard_split_split(
    input: SplitPrimitiveTensor, *, dim: int, count: int, devices: None
) -> SplitPrimitiveTensor:
    dim = normalize_negative_dim(input, dim)
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    if input.shard_dim != dim:
        raise ValueError(f"Resharding is not supported")
    return input


@reshard_split.override(ReplicatedTensor)
def reshard_split_replicated(
    input: ReplicatedTensor, *, dim: int, count: int, devices: None
) -> SplitPrimitiveTensor:
    dim = normalize_negative_dim(input, dim)
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    if input.shape[dim] % count != 0:
        raise ValueError(
            f"Split resharding with uneven splits not supported."
            f" Dimension size {input.shape[dim]} must be divisible by"
            f" {count}"
        )

    assert (
        input.shape[dim] >= count
    ), f"Cannot split dimension {dim} of size {input.shape[dim]} into {count} shards"

    def slice_range_along_dim(dim: int, start: int, end: int):
        res = [slice(None)] * len(input.shape)
        res[dim] = slice(start, end)
        return res

    shard_size_along_dim = input.shape[dim] // count
    shards = [
        unbox_tensor(shard)[
            slice_range_along_dim(
                dim=dim,
                start=shard_idx * shard_size_along_dim,
                end=(shard_idx + 1) * shard_size_along_dim,
            )
        ]
        for shard_idx, shard in enumerate(input.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=dim, devices=input.devices)


@reshard_like.override(Tensor, Tensor)
def reshard_like_unsharded_to_unsharded(input, like: Tensor) -> Tensor:
    return input


@reshard_like.override(Tensor, SplitPrimitiveTensor)
def reshard_like_unsharded_to_split(
    input, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return reshard_split(torch_input, dim=like.shard_dim, count=like.shard_count)


@reshard_like.override(ReplicatedTensor, Tensor)
def reshard_like_replicated_to_unsharded(input: ReplicatedTensor, like):
    return input.shards[0]


@reshard_like.override(SplitPrimitiveTensor, Tensor)
def reshard_like_split_to_unsharded(input: SplitPrimitiveTensor, like):
    return sharded_cat(input)


@reshard_like.override(Tensor, ReplicatedTensor)
def reshard_like_unsharded_to_replicated(
    tensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    torch_tensor = unbox_tensor(tensor)
    return replicate(torch_tensor, count=like.shard_count, devices=like.devices)


@reshard_like.override(ReplicatedTensor, ReplicatedTensor)
def reshard_like_replicated_to_replicated(
    tensor: ReplicatedTensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    if tensor.shard_count != like.shard_count:
        raise ValueError(
            f"Operands' number of shards not equal ({input.shard_count} != {like.shard_count})"
        )
    return tensor


@reshard_like.override(ReplicatedTensor, SplitPrimitiveTensor)
def reshard_like_replicated_to_split(
    tensor: ReplicatedTensor, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    """
    Adjust to handle broadcasting.
    If `like` has more dims than `tensor`, we meed to decrease dim by the difference.
    If it has more dims we need to increase dim instead.
    Conceptually we are right aligning the dims.
      like.shape     == [1, 2, 3]
      tensor.shape   == [2, 3]
    Becomes:
      like.shape     == [1, 2, 3]
      tensor.shape   == [   2, 3]
    """
    dim = (
        like.shard_dim
        - max(0, len(like.shape) - len(tensor.shape))
        + max(0, len(tensor.shape) - len(like.shape))
    )
    return reshard_split(tensor, dim=dim, count=like.shard_count)


@reshard_like.override(SplitPrimitiveTensor, ReplicatedTensor)
def reshard_like_split_to_replicated(
    tensor: SplitPrimitiveTensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    return all_gather(tensor)


@reshard_like.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def reshard_like_split_to_split(
    tensor: SplitPrimitiveTensor, like: SplitPrimitiveTensor
) -> SplitPrimitiveTensor:
    assert (
        tensor.shard_count == like.shard_count and tensor.shard_dim == like.shard_dim
    ), "Resharding is not supported"
    return tensor


@reshard_like.override(UnreducedTensor, ReplicatedTensor)
def reshard_like_unreduced_to_replicated(
    tensor: UnreducedTensor, like: ReplicatedTensor
) -> ReplicatedTensor:
    return replicate(tensor, count=like.shard_count)


@scatter_.override(SplitPrimitiveTensor, SplitPrimitiveTensor, Number)
def scatter_split_split(
    inout: SplitPrimitiveTensor,
    dim: int,
    index: SplitPrimitiveTensor,
    value: Number,
    *,
    reduce: str = None,
) -> SplitPrimitiveTensor:
    assert isinstance(value, Number), "Tensor version of this op not implemented"
    if dim == inout.shard_dim:
        # `index` can contain indices into any of `inout`s shards in any of its entries.
        # Can't know ahead of time how to seperate out its values based on sliices.
        tmp_tensor = all_gather(inout)
        index = all_gather(index)
        tmp_tensor.scatter_(dim, index, value, reduce=reduce)
        tmp_tensor = reshard_like(tmp_tensor, inout)

        for inout_shard, tmp_shard in zip(inout.shards, tmp_tensor.shards):
            inout_shard.as_torch().copy_(tmp_shard.as_torch())
        return inout

    shard_dim = inout.shard_dim
    if index.shape[shard_dim] == inout.shape[shard_dim]:
        assert index.shard_dim == inout.shard_dim
        index_shards = index.shards
        last_shard_idx = inout.shard_count - 1
    else:
        # If the shapes are not the same it means that:
        #   1. Not all slices along dim inside `inout` will be accessed (so we can decrease computation)
        #   2. Slices indo shards of `index` and `inout` will not line up,
        #      i.e. The slice index_shard_i[j] will not match up to inout_shard_i[j]
        index = all_gather(index)

        # Find the last shard of `inout` that will be accessed.
        slice_indices_inout = [shard.shape[shard_dim] for shard in inout.shards]
        cumulative_slice_idx = list(itertools.accumulate(slice_indices_inout))
        final_slice_idx = index.shards[0].shape[shard_dim]  # Replicated, all the same
        last_shard_idx = max(
            i for i, val in enumerate(cumulative_slice_idx) if val <= final_slice_idx
        )

        # Manually re-shard and re-scatter index
        # NOTE: index may not have the same number of shards as inout.
        size_along_shard_dim = []
        num_slices_left = final_slice_idx
        for i in range(last_shard_idx + 1):
            size_along_shard_dim.append(min(num_slices_left, slice_indices_inout[i]))
            num_slices_left -= size_along_shard_dim[-1]
        assert num_slices_left == 0
        index_shards = unbox_tensor(index).split(size_along_shard_dim, dim=shard_dim)
        index_shards = [
            transfer_to_logical_device(shard, index.devices[i])
            for i, shard in enumerate(index_shards)
        ]
        assert len(index_shards) == last_shard_idx + 1

    for i in range(last_shard_idx + 1):
        inout.shards[i].scatter_(
            dim,
            unbox_tensor(index_shards[i]),
            value,
            reduce=reduce,
        )

    return inout


@sharded_cat.override(SplitPrimitiveTensor)
def sharded_cat_unsharded(tensor: SplitPrimitiveTensor) -> InferenceTensor:
    shard_ts = [
        (
            transfer_to_logical_device(shard, tensor.devices[0])
            if i != 0
            else barrier_on_logical_device(shard, tensor.devices[0])
        )
        for i, shard in enumerate(tensor.shards)
    ]
    return cat(shard_ts, dim=tensor.shard_dim)


@sharded_gather.override(IsOfType(SplitPrimitiveTensor, ReplicatedTensor))
def sharded_gather_split(
    input: SplitPrimitiveTensor | ReplicatedTensor, root_rank: int
) -> List[Tensor]:
    # if input is SplitPrimitiveTensor
    if type(input) == SplitPrimitiveTensor:
        shard_ts = [
            (
                transfer_to_logical_device(shard, input.devices[root_rank])
                if i != root_rank
                else barrier_on_logical_device(shard, input.devices[root_rank])
            )
            for i, shard in enumerate(input.shards)
        ]
        return shard_ts
    else:
        shard = input.shards[root_rank]
        return [shard.as_torch().clone() for _ in range(input.shard_count)]


@shards.override(BlockScaledLayout)
def shards_split_quantized_layout(input: BlockScaledLayout) -> list[BlockScaledLayout]:
    if not all(isinstance(v, SplitPrimitiveTensor) for v in input.planes.values()):
        return NotImplemented

    block_shape = [i // d for i, d in zip(input.shape, input.d.shape[:-1], strict=True)]
    shard_layout_shapes = [
        [
            d_shape_dim * block_shape_dim
            for d_shape_dim, block_shape_dim in zip(
                d_shard.shape[:-1], block_shape, strict=True
            )
        ]
        for d_shard in input.d.shards
    ]

    def get_plane_shards(
        planes: dict[str, SplitPrimitiveTensor], shard_idx: int
    ) -> dict[str, AnyTensor]:
        return {name: tensor.shards[shard_idx] for name, tensor in planes.items()}

    return [
        input.create(
            shape=shape,
            metadata=input.metadata,
            planes=get_plane_shards(input.planes, i),
        )
        for i, shape in enumerate(shard_layout_shapes)
    ]


@shards.override(ShardedTensor)
def shards_sharded_tensor(input: ShardedTensor) -> list[AnyTensor]:
    return input.shards


def _sharded_sum_sharded(tensor: ShardedTensor, root_rank: int) -> Tensor:
    if root_rank < 0 or root_rank >= tensor.shard_count:
        raise ValueError(
            f"Root rank {root_rank} must be in the range [0, {tensor.shard_count})"
        )
    reduced = functools.reduce(
        lambda x, y: elementwise(torch.add, x, y),
        [
            (
                transfer_to_logical_device(shard, tensor.devices[root_rank])
                if i != root_rank
                else barrier_on_logical_device(shard, tensor.devices[root_rank])
            )
            for i, shard in enumerate(tensor.shards)
        ],
    )
    return reduced


@sharded_sum.override(IsOfType(SplitPrimitiveTensor, UnreducedTensor))
def sharded_sum_split(
    input: SplitPrimitiveTensor | UnreducedTensor, root_rank: int = 0
) -> Tensor:
    return _sharded_sum_sharded(input, root_rank)


@sigmoid.override(ShardedTensor)
def sigmoid_sharded(tensor: ShardedTensor) -> ShardedTensor:
    return elementwise(torch.sigmoid, tensor)


@softmax.override(SplitPrimitiveTensor)
def softmax_split(
    tensor: SplitPrimitiveTensor, dim: Optional[int], dtype: Optional[torch.dtype]
) -> Tensor:
    dim = dim if dim is None or dim >= 0 else len(tensor.shape) + dim
    assert (
        dim is not None and dim != tensor.shard_dim
    ), "Softmax along split dimension is not supported."
    shards = [softmax(shard, dim=dim, dtype=dtype) for shard in tensor.shards]
    return SplitPrimitiveTensor(
        ts=shards, shard_dim=tensor.shard_dim, shape=tensor.shape
    )


@split.override(UnreducedTensor)
def split_unreduced(
    tensor: UnreducedTensor, split_size_or_sections: int | list[int], dim: int = 0
) -> tuple[UnreducedTensor, ...]:
    # Example of splitting in 3 pieces a tensor distributed over 2
    # devices.
    # Device placement before split:
    # +---+ +---+
    # |   | |   |
    # |   | |   |
    # | 0 | | 1 |
    # |   | |   |
    # |   | |   |
    # +---+ +---+
    #
    # after split:
    # +---+ +---+
    # | 0 | | 1 | <- shards of result tensor 0
    # |---| |---|
    # | 0 | | 1 | <- shards of result tensor 1
    # |---| |---|
    # | 0 | | 1 | <- shards of result tensor 2
    # +---+ +---+
    #
    # No transfering is required, just reinterpretation of the pieces.

    splits_per_shard = [
        split(shard, split_size_or_sections, dim) for shard in tensor.shards
    ]
    # transpose nested list of lists.
    shards_per_split = list(zip(*splits_per_shard, strict=True))
    return [UnreducedTensor(ts=shards) for shards in shards_per_split]


@sum.override(SplitPrimitiveTensor)
def sum_split(
    input: SplitPrimitiveTensor,
    dim: int | List[int] | None,
    keepdim: bool,
    *,
    dtype: torch.dtype,
) -> SplitPrimitiveTensor | ReplicatedTensor:
    assert dim is not None, "sum dim must be specified"
    if not isinstance(dim, (list, tuple)):
        dim = [dim]
    # Handle negative indexing
    dim = [d + len(input.shape) if d < 0 else d for d in dim]

    if input.shard_dim not in dim:
        shard_dim = input.shard_dim
        # Have to offest `shard_dim` if any of the collapsing dims are "to the left of it".
        if not keepdim:
            # `sum` is clobbered by ops.sum, need to access it manually
            shard_dim -= sum(d < input.shard_dim for d in dim)

        shards = [
            sum(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in input.shards
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)
    else:
        gathered = cat(
            [
                (
                    transfer_to_logical_device(shard, input.devices[0])
                    if i != 0
                    else barrier_on_logical_device(shard, input.devices[0])
                )
                for i, shard in enumerate(input.shards)
            ],
            dim=input.shard_dim,
        )
        summed = sum(gathered, dim=dim, keepdim=keepdim, dtype=dtype)
        return ReplicatedTensor(ts=summed, shard_count=input.shard_count)


@to.override(ShardedTensor)
def to_sharded(tensor: ShardedTensor, *args, **kwargs):
    shards = [to(shard, *args, **kwargs) for shard in tensor.shards]
    return tensor.clone(ts=shards)


@topk.override(SplitPrimitiveTensor)
def topk_split(
    input: SplitPrimitiveTensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    use_linalgext_topk: bool,
) -> tuple[
    SplitPrimitiveTensor | ReplicatedTensor, SplitPrimitiveTensor | ReplicatedTensor
]:
    if dim != input.shard_dim:
        values, indices = zip(
            *(
                topk(
                    shard,
                    k=k,
                    dim=dim,
                    largest=largest,
                    sorted=sorted,
                    use_linalgext_topk=use_linalgext_topk,
                )
                for shard in input.shards
            )
        )
        values_split = SplitPrimitiveTensor(ts=values, shard_dim=input.shard_dim)
        indices_split = SplitPrimitiveTensor(ts=indices, shard_dim=input.shard_dim)
        return values_split, indices_split
    else:
        # TODO: implement using all_reduce_topk when IREE supports it

        all_v_loc = []
        all_i_glob = []
        offset = 0
        for i, shard in enumerate(input.shards):
            v_loc, i_loc = topk(shard, k=k, dim=dim, largest=largest, sorted=sorted)

            i_glob = i_loc + offset
            offset += shard.shape[dim]

            if i == 0:
                v_loc = barrier_on_logical_device(v_loc, input.devices[0])
                i_glob = barrier_on_logical_device(i_glob, input.devices[0])
            else:
                v_loc = transfer_to_logical_device(v_loc, input.devices[0])
                i_glob = transfer_to_logical_device(i_glob, input.devices[0])

            all_v_loc.append(v_loc)
            all_i_glob.append(i_glob)

        cat_i_glob = cat(all_i_glob, dim=dim)
        cat_v_loc = cat(all_v_loc, dim=dim)

        total_vals, pos = topk(cat_v_loc, k=k, dim=dim, largest=largest, sorted=sorted)
        total_inds = torch.take_along_dim(cat_i_glob, pos, dim=dim)

        top_vals = ReplicatedTensor(ts=total_vals, shard_count=input.shard_count)
        top_inds = ReplicatedTensor(ts=total_inds, shard_count=input.shard_count)

        return top_vals, top_inds


@unpack.override(SplitPrimitiveTensor)
def unpack_split(input: SplitPrimitiveTensor) -> QuantizedLayout:
    layouts = [unpack(shard) for shard in input.shards]
    planes_per_leayout = [layout.planes for layout in layouts]

    shards_per_plane = tree.map_leaves(
        planes_per_leayout[0], f=lambda x: [], is_leaf=is_any_tensor
    )

    def reduce_fn(value: list[AnyTensor], tensor: AnyTensor) -> list[AnyTensor]:
        value.append(tensor)
        return value

    shards_per_plane = tree.reduce_horizontal(
        fn=reduce_fn,
        trees=planes_per_leayout,
        initial=shards_per_plane,
        is_leaf=is_any_tensor,
    )

    def make_sharded_tensor(shards: list[AnyTensor]) -> ShardedTensor:
        if len(shards[0].shape) == 0:
            return ReplicatedTensor(ts=shards, devices=input.devices)
        else:
            return SplitPrimitiveTensor(
                ts=shards, devices=input.devices, shard_dim=input.shard_dim
            )

    sharded_planes = {
        name: make_sharded_tensor(shards) for name, shards in shards_per_plane.items()
    }
    metadata = layouts[0].metadata
    for layout in layouts[1:]:
        tree.assert_equal(metadata, layout.metadata)
    return type(layouts[0]).create(
        shape=input.shape, metadata=metadata, planes=sharded_planes
    )


@unpack_qs.override(SplitPrimitiveTensor, BlockScaledFp4Layout)
def unpack_qs_split_block_scaled_fp4_layout(
    qs: SplitPrimitiveTensor, layout: BlockScaledFp4Layout
) -> SplitPrimitiveTensor:
    layout_per_shard = shards(layout)
    result_shards = [
        unpack_qs(qs_shard, shard_layout)
        for qs_shard, shard_layout in zip(qs.shards, layout_per_shard, strict=True)
    ]
    return SplitPrimitiveTensor(
        ts=result_shards, shard_dim=qs.shard_dim, devices=qs.devices
    )


@unshard.override(ReplicatedTensor)
def unshard_replicated(input: ReplicatedTensor) -> InferenceTensor:
    return input.shards[0]


@unshard.override(SplitPrimitiveTensor)
def unshard_split(input: SplitPrimitiveTensor) -> InferenceTensor:
    return sharded_cat(input)


@unshard.override(QuantizedLayout)
def unshard_layout(layout: QuantizedLayout) -> QuantizedLayout:
    unsharded_planes = {
        name: unbox_tensor(unshard(plane)) for name, plane in layout.planes.items()
    }
    return type(layout).create(
        shape=layout.shape, metadata=layout.metadata, planes=unsharded_planes
    )


@unshard.override(UnreducedTensor)
def unshard_unreduced(input: UnreducedTensor) -> InferenceTensor:
    shards = input.shards
    shards = [
        (
            barrier_on_logical_device(shard, input.devices[0])
            if i == 0
            else transfer_to_logical_device(shard, input.devices[0])
        )
        for i, shard in enumerate(shards)
    ]
    return functools.reduce(lambda x, y: elementwise(torch.add, x, y), shards)


@unshard.override(Tensor)
def unshard_unsharded(input: Tensor) -> Tensor:
    return input


@view_as_complex.override(SplitPrimitiveTensor)
def view_as_complex_split(tensor: SplitPrimitiveTensor) -> SplitPrimitiveTensor:
    shards = [view_as_complex(shard) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@view_as_real.override(SplitPrimitiveTensor)
def view_as_real_split(tensor: SplitPrimitiveTensor) -> SplitPrimitiveTensor:
    shards = [view_as_real(shard) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@zeros_like.override(AllOfType(ReplicatedTensor, SplitPrimitiveTensor))
def zeros_like_replicated(
    tensor: ReplicatedTensor | SplitPrimitiveTensor,
    *,
    dtype: torch.dtype | None,
    layout: torch.layout | None,
    device: torch.device | None,
    requires_grad: bool,
    memory_format: torch.memory_format,
) -> ReplicatedTensor | SplitPrimitiveTensor:
    shards = [
        zeros_like(
            shard,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )
        for shard in tensor.shards
    ]
    return tensor.clone(ts=shards)


# Note: Must be last thing in file
sharded_unwrap_override()
