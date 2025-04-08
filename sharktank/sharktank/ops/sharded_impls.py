# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union, Any, Tuple, Dict, Iterable
import itertools
from numbers import Number
import math
import functools

from ..types import (
    AnyTensor,
    DefaultPrimitiveTensor,
    InferenceTensor,
    PrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    sharding,
    SplitPrimitiveTensor,
    Theta,
    UnreducedTensor,
)
from ..types.tensors import unbox_tensor
from ._registry import AllOfType, AllOfExprsVariadic, IsOfType
from .shape import broadcast_dims, broadcast_dim, unbroadcast_dim
from ..utils import longest_equal_range
from .signatures import *


def assert_on_same_devices(*tensors: Tuple[ShardedTensor]) -> None:
    """
    Checks that all tensors are placed on the same devices.
    """
    if len(tensors) <= 1:
        return
    assert all(isinstance(tensor, ShardedTensor) for tensor in tensors)

    for tensor in tensors[1:]:
        if any(d0 != d for d0, d in zip(tensors[0].devices, tensor.devices)):
            raise ValueError("All tensors must be placed on the same devices.")


def sharded_wrap_override():
    def transfer_n_pin(f):
        """
        Wrapper for each NON-TRANSFERING op defined in this file.
        """

        def func_wrapper(*args: Tuple, **kwargs: Dict[str, Any]):
            """
            Wraps each NON-TRANSFERING operation, f, to ensure that all incoming tensors are on the same device and that the result has the devices correctly labelled.

            If no ShardedTensors are present in the input, then no changes are made to input/output.
            """
            sharded_tensors = []
            for value in itertools.chain(args, kwargs.values()):
                if isinstance(value, ShardedTensor):
                    sharded_tensors.append(value)
                    continue
                if isinstance(value, Iterable):
                    for val in value:
                        if isinstance(val, ShardedTensor):
                            sharded_tensors.append(val)

            assert_on_same_devices(*sharded_tensors)
            res = f(*args, **kwargs)
            if isinstance(res, ShardedTensor) and len(sharded_tensors) > 0:
                res = res.clone(devices=sharded_tensors[0].devices)
            return res

        return func_wrapper

    def wrap_override(signature_dispatcher_override):
        """
        Wrap [op].override's result so that the transfer_n_pin(f) becomes the target in _TargetOverride rather than f itself.
        """

        def override_return_wrapper(*override_args, **override_kwargs):
            orig_decorator = signature_dispatcher_override(
                *override_args, **override_kwargs
            )
            new_decorator = lambda f: orig_decorator(transfer_n_pin(f))
            return new_decorator

        return override_return_wrapper

    do_not_wrap = {
        "all_gather",
        "all_reduce",
        "replicate",
        "index_copy_",
        "index_put_",
        "transfer_to_logical_device",
        "reshard_like",
        "replicate_like",
        "equal",
    }

    from . import signatures

    for func_name in signatures.__all__:
        func = globals()[func_name]
        if (func_name not in do_not_wrap) and (hasattr(func, "override")):
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


sharded_wrap_override()


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


@cat.override(AllOfType(ReplicatedTensor))
def cat_replicated(tensors: Sequence[ReplicatedTensor], dim: int) -> ReplicatedTensor:
    assert len(tensors) > 0
    shard_count = tensors[0].shard_count
    assert all([t.shard_count == shard_count for t in tensors])

    shards = [cat(shards, dim) for shards in zip(*[t.shards for t in tensors])]
    return ReplicatedTensor(ts=shards)


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

# Sharded elementwise.


@elementwise.override(ReplicatedTensor)
def replicated_elementwise_unary(operator, x: ReplicatedTensor, *args, **kwargs):
    partials = [operator(unbox_tensor(pt), *args, **kwargs) for pt in x.shards]
    return ReplicatedTensor(ts=partials)


@elementwise.override(SplitPrimitiveTensor)
def split_elementwise_unary(operator, x: SplitPrimitiveTensor, *args, **kwargs):
    partials = [operator(unbox_tensor(pt), *args, **kwargs) for pt in x.shards]
    return SplitPrimitiveTensor(shard_dim=x.shard_dim, shape=x.shape, ts=partials)


@elementwise.override(ReplicatedTensor, ReplicatedTensor)
def replicated_elementwise_binary(
    operator, x: ReplicatedTensor, y: ReplicatedTensor, *args, **kwargs
):
    assert x.shard_count == y.shard_count
    shards = [
        operator(unbox_tensor(shard_x), unbox_tensor(shard_y), *args, **kwargs)
        for shard_x, shard_y in zip(x.shards, y.shards)
    ]
    return ReplicatedTensor(ts=shards)


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
        shard_dim=x.shard_dim,
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


# Embedding Lookup
@embedding_lookup.override(ReplicatedTensor, ReplicatedTensor)
def embedding_lookup_default(
    input: ReplicatedTensor,
    embedding_matrix: ReplicatedTensor,
    dtype: Optional[torch.dtype],
):
    assert input.shard_count == embedding_matrix.shard_count
    shards = [
        embedding_lookup(input_shard, embedding_matrix_shard, dtype)
        for input_shard, embedding_matrix_shard in zip(
            input.shards, embedding_matrix.shards
        )
    ]
    return ReplicatedTensor(ts=shards)


@expand.override(ReplicatedTensor)
def expand_replicated(tensor: ReplicatedTensor, shape: List[int]) -> ReplicatedTensor:
    shards = [expand(shard, shape) for shard in tensor.shards]
    return tensor.clone(ts=shards)


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


@flatten.override(ReplicatedTensor)
def flatten_replicated(
    input: ReplicatedTensor, start_dim: int, end_dim: int
) -> ReplicatedTensor:
    shards = [shard.flatten(start_dim, end_dim) for shard in input.shards]
    return ReplicatedTensor(ts=shards)


@flatten.override(SplitPrimitiveTensor)
def flatten_split(
    input: SplitPrimitiveTensor, start_dim: int, end_dim: int
) -> SplitPrimitiveTensor:
    end_dim_resolved = len(input.shape) - 1 if end_dim == -1 else end_dim
    assert input.shard_dim <= start_dim or end_dim_resolved < input.shard_dim, (
        "Flattening of a sharded dimension that is not the leading dimension in the"
        " flattening dimension range is not supported. This would result in a"
        " block-cyclic sharding which is not implemented."
    )
    assert (
        input.shard_dim != start_dim
        or input.shape[input.shard_dim] % input.shard_count == 0
    ), "If the leading flattening dimension is the split dimension, its size must be divisible by the shard count."
    shards = [shard.flatten(start_dim, end_dim) for shard in input.shards]
    shard_dim = (
        input.shard_dim
        if input.shard_dim <= start_dim
        else input.shard_dim - (end_dim_resolved - start_dim)
    )
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@gather.override(ReplicatedTensor, ReplicatedTensor)
def gather_replicated(
    input: ReplicatedTensor, dim: int, index: ReplicatedTensor
) -> Tensor:
    assert input.shard_count == index.shard_count
    shards = [
        gather(input_shard, dim, index_shard)
        for input_shard, index_shard in zip(input.shards, index.shards)
    ]
    return ReplicatedTensor(ts=shards)


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


@index_copy_.override(ReplicatedTensor, ReplicatedTensor, ReplicatedTensor)
def index_copy__replicated_replicated_replicated(
    inout: ReplicatedTensor,
    dim: int,
    index: ReplicatedTensor,
    tensor: ReplicatedTensor,
) -> ReplicatedTensor:
    assert (
        inout.shard_count == index.shard_count
        and inout.shard_count == tensor.shard_count
    )
    for inout_shard, index_shard, tensor_shard in zip(
        inout.shards, index.shards, tensor.shards
    ):
        index_copy_(inout_shard, dim, index_shard, tensor_shard)
    return inout


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


@index_select.override(ReplicatedTensor, ReplicatedTensor)
def index_select_replicated(
    tensor: ReplicatedTensor,
    dim: int,
    index: ReplicatedTensor,
) -> ReplicatedTensor:
    assert tensor.shard_count == index.shard_count
    shards = [
        index_select(tensor_shard, dim, index_shard)
        for tensor_shard, index_shard in zip(tensor.shards, index.shards)
    ]
    return ReplicatedTensor(ts=shards)


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


@interpolate.override(ReplicatedTensor)
def interpolate_replicated(
    input: ReplicatedTensor,
    size: Optional[int | List[int]],
    scale_factor: Optional[float | List[float]],
    mode: str,
    align_corners: Optional[bool],
    recompute_scale_factor: Optional[bool],
    antialias: bool,
) -> ReplicatedTensor:
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
    return ReplicatedTensor(ts=shards)


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
) -> SplitPrimitiveTensor:
    # TODO: handle different dtypes
    result = matmul(input, weight.T)
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


# Sharded matmuls.


@matmul.override(ReplicatedTensor, ReplicatedTensor)
def matmul_replicated(
    lhs: ReplicatedTensor, rhs: ReplicatedTensor, *, transpose_rhs: bool
) -> ReplicatedTensor:
    assert lhs.shard_count == rhs.shard_count

    if transpose_rhs:
        return matmul(lhs, rhs.T)

    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return ReplicatedTensor(ts=shards)


@matmul.override(ReplicatedTensor, SplitPrimitiveTensor)
def matmul_replicated_lhs_split_rhs(
    lhs: ReplicatedTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> SplitPrimitiveTensor | UnreducedTensor:
    assert lhs.shard_count == rhs.shard_count
    assert len(rhs.shape) == 2

    if transpose_rhs:
        return matmul(lhs, rhs.T)

    rhs_reduction_dim = 1
    if rhs_reduction_dim != rhs.shard_dim:
        lhs_reduction_dimension = len(lhs.shape) - 1
        lhs_split = reshard_split(
            lhs, dim=lhs_reduction_dimension, count=lhs.shard_count
        )
        return matmul(lhs_split, rhs)

    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=len(lhs.shape) - 2 + rhs.shard_dim)


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
    assert lhs_reduction_dim != lhs.shard_dim
    if transpose_rhs:
        rhs = rhs.T
    shards = [
        matmul(lhs_shard, rhs_shard)
        for (lhs_shard, rhs_shard) in zip(lhs.shards, rhs.shards)
    ]
    return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)


@matmul.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
def matmul_split(
    lhs: SplitPrimitiveTensor, rhs: SplitPrimitiveTensor, *, transpose_rhs: bool
) -> UnreducedTensor | SplitPrimitiveTensor:
    if lhs.shard_count != rhs.shard_count:
        raise ValueError(
            f"Cannot matmul split tensors of different shard_count: "
            f"({lhs.shard_count} vs {rhs.shard_count})"
        )
    if transpose_rhs:
        return matmul(lhs, rhs.T)

    lhs_reduction_dim = len(lhs.shape) - 1
    rhs_reduction_dim = len(rhs.shape) - 2 if len(rhs.shape) > 1 else len(rhs.shape) - 1

    # The reduction dimension is split on both tensors.
    if lhs_reduction_dim == lhs.shard_dim and rhs_reduction_dim == rhs.shard_dim:
        partials = [
            matmul(partial_lhs, partial_rhs)
            for partial_lhs, partial_rhs in zip(lhs.shards, rhs.shards)
        ]
        return UnreducedTensor(ts=partials)

    is_batched_matmul = len(lhs.shape) > 2 or len(rhs.shape) > 2
    if (
        is_batched_matmul
        and len(lhs.shape) == len(rhs.shape)
        and lhs.shard_dim == rhs.shard_dim
    ):
        # The same batch dim is sharded for both arguments.
        shards = [
            matmul(lhs_shard, rhs_shard)
            for lhs_shard, rhs_shard in zip(lhs.shards, rhs.shards)
        ]
        return SplitPrimitiveTensor(ts=shards, shard_dim=lhs.shard_dim)

    # -1 for missing parallel dim.
    lhs_parallel_dim = len(lhs.shape) - 2
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
        return matmul(lhs, replicated_rhs)

    assert False, "Sharding configuration not supported"


# Scaled dot product attention
@scaled_dot_product_attention.override(
    ReplicatedTensor, ReplicatedTensor, ReplicatedTensor, Optional[ReplicatedTensor]
)
def scaled_dot_product_attention_replicated(
    q: ReplicatedTensor,
    k: ReplicatedTensor,
    v: ReplicatedTensor,
    a: Optional[ReplicatedTensor],
    is_causal: bool,
    scale: float,
) -> ReplicatedTensor:
    if q.shard_count != k.shard_count or q.shard_count != v.shard_count:
        raise ValueError("Incompatible number of shards for qkv")

    if a and q.shard_count != a.shard_count:
        raise ValueError(
            f"Incompatible number of shards for a ({a.shard_count}) should be ({q.shard_count})"
        )
    a_shards = [None] * q.shard_count if a is None else a.shards

    output_shards = []
    for q_s, k_s, v_s, a_s in zip(q.shards, k.shards, v.shards, a_shards):
        o_s = scaled_dot_product_attention(
            q_s, k_s, v_s, a_s, is_causal=is_causal, scale=scale
        )
        output_shards.append(o_s)

    return ReplicatedTensor(ts=output_shards)


@scaled_dot_product_attention.override(
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    SplitPrimitiveTensor,
    Optional[ReplicatedTensor],
)
def scaled_dot_product_attention_sharded(
    q, k, v, a, is_causal, scale
) -> SplitPrimitiveTensor:
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
            q_s, k_s, v_s, a_s, is_causal=is_causal, scale=scale
        )
        output_shards.append(o_s)

    return SplitPrimitiveTensor(ts=output_shards, shard_dim=q.shard_dim)


@mean.override(ReplicatedTensor)
def mean_replicated(
    x: ReplicatedTensor,
    dim: Union[int, List[int]],
    keepdim: bool,
    *,
    dtype: torch.dtype,
) -> None:
    shards = [mean(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in x.shards]
    return ReplicatedTensor(ts=shards)


@module_register_buffer.override(torch.nn.Module, ShardedTensor)
def module_register_buffer_sharded(
    module: torch.nn.Module, name: str, tensor: ShardedTensor
) -> None:
    for i, shard in enumerate(tensor.shards):
        module_register_buffer(module, f"{name}__shard__{i}", shard)
    setattr(module, name, tensor)


@permute.override(SplitPrimitiveTensor)
def permute_split(tensor: SplitPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return SplitPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)


@permute.override(ReplicatedTensor)
def permute_replicated(tensor: ReplicatedTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    return ReplicatedTensor(ts=permuted_shards)


@repeat.override(ReplicatedTensor)
def repeat_replicated(input: ReplicatedTensor, *sizes: List[int]) -> ReplicatedTensor:
    shards = [repeat(shard, *sizes) for shard in input.shards]
    return ReplicatedTensor(ts=shards)


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


@reshape.override(SplitPrimitiveTensor)
def reshape_split(
    tensor: SplitPrimitiveTensor, shape: List[int]
) -> SplitPrimitiveTensor:
    if _reshape_get_single_split_dim(tensor.shape, shape) is not None:
        return view(tensor, shape)

    flatten_dim_range = _reshape_get_flatten_dim_range(tensor.shape, shape)
    if flatten_dim_range is not None:
        return flatten(tensor, flatten_dim_range[0], flatten_dim_range[1] - 1)

    raise ValueError(
        f"Unsupported reshaping of sharded split tensor of shape {tensor.shape} to shape {shape}"
    )


@reshard.override(Tensor, sharding.Split)
def reshard_tensor_split(input: Tensor, spec: sharding.Split) -> AnyTensor:
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


@reshard_split.override(Tensor)
def reshard_split_unsharded(
    input, *, dim: int, count: int, devices: tuple[int, ...]
) -> SplitPrimitiveTensor:
    torch_input = unbox_tensor(input)
    return SplitPrimitiveTensor(
        ts=torch_input, shard_dim=dim, shard_count=count, devices=devices
    )


@reshard_split.override(SplitPrimitiveTensor)
def reshard_split_split(
    input: SplitPrimitiveTensor, *, dim: int, count: int, devices: None
) -> SplitPrimitiveTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")
    if input.shard_dim != dim:
        raise ValueError(f"Resharding is not supported")
    return input


@reshard_split.override(ReplicatedTensor)
def reshard_split_replicated(
    input: ReplicatedTensor, *, dim: int, count: int, devices: None
) -> SplitPrimitiveTensor:
    if input.shard_count != count:
        raise ValueError(f"Number of shards not equal ({input.shard_count} != {count})")

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
    return reshard_split(tensor, dim=like.shard_dim, count=like.shard_count)


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


@sharded_cat.override(SplitPrimitiveTensor)
def sharded_cat_unsharded(tensor: SplitPrimitiveTensor):
    shard_ts = [
        (
            transfer_to_logical_device(shard.as_torch(), tensor.devices[0])
            if i != 0
            else barrier_on_logical_device(shard.as_torch(), tensor.devices[0])
        )
        for i, shard in enumerate(tensor.shards)
    ]
    return torch.cat(shard_ts, dim=tensor.shard_dim)


# Sharded sum.


def _sharded_sum_sharded(tensor: ShardedTensor) -> Tensor:
    accum = tensor.shards[0].as_torch()
    for shard in tensor.shards[1:]:
        accum = torch.add(accum, shard.as_torch())
    return accum


@sharded_sum.override(SplitPrimitiveTensor)
def sharded_sum_split(maybe_sharded: SplitPrimitiveTensor) -> Tensor:
    # TODO: Should implement as an all reduce.
    return _sharded_sum_sharded(maybe_sharded)


@sharded_sum.override(UnreducedTensor)
def sharded_sum_unreduced(maybe_sharded: UnreducedTensor) -> Tensor:
    return _sharded_sum_sharded(maybe_sharded)


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


@to.override(ReplicatedTensor)
def to_replicated(tensor: ReplicatedTensor, *args, **kwargs):
    shards = [to(shard, *args, **kwargs) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@to.override(SplitPrimitiveTensor)
def to_split(tensor: SplitPrimitiveTensor, *args, **kwargs):
    shards = [to(shard, *args, **kwargs) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@transpose.override(ReplicatedTensor)
def transpose_replicated(
    tensor: ReplicatedTensor, dim0: int, dim1: int
) -> ReplicatedTensor:
    shards = [transpose(shard, dim0, dim1) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@transpose.override(SplitPrimitiveTensor)
def transpose_split(
    tensor: SplitPrimitiveTensor, dim0: int, dim1: int
) -> SplitPrimitiveTensor:
    shards = [transpose(shard, dim0, dim1) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    if shard_dim == dim0:
        shard_dim = dim1
    elif shard_dim == dim1:
        shard_dim = dim0
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unflatten.override(ReplicatedTensor)
def unflatten_replicated(
    input: ReplicatedTensor, dim: int, sizes: Tuple[int]
) -> ReplicatedTensor:
    shards = [unflatten(shard, dim, sizes) for shard in input.shards]
    return input.clone(ts=shards)


@unflatten.override(SplitPrimitiveTensor)
def unflatten_split(
    input: SplitPrimitiveTensor, dim: int, sizes: Tuple[int]
) -> SplitPrimitiveTensor:
    if dim == input.shard_dim:
        if sizes[0] == -1:
            assert (
                dim != input.shard_dim
            ), "Unflattening the split dimension is not supported."
        sizes = tuple([sizes[0] // input.shard_dim] + [s for s in sizes[1:]])
    shards = [unflatten(shard, dim, sizes) for shard in input.shards]
    shard_dim = input.shard_dim
    if dim < shard_dim:
        shard_dim += len(sizes) - 1
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unshard.override(ReplicatedTensor)
def unshard_replicated(input: ReplicatedTensor) -> Tensor:
    return input.shards[0]


@unshard.override(SplitPrimitiveTensor)
def unshard_split(input: SplitPrimitiveTensor) -> Tensor:
    return sharded_cat(input)


@unshard.override(UnreducedTensor)
def unshard_unreduced(input: UnreducedTensor) -> Tensor:
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


def _reshape_get_flatten_dim_range(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would flatten a range of dimensions return that index range [begin, end).
    If the reshape is not of that kind return `None`."""
    flatten_start_len = _reshape_get_single_split_dim(to_shape, from_shape)
    if flatten_start_len is None:
        return None
    start, length = flatten_start_len
    return start, start + length


def _reshape_infer_dynamic_dim(
    shape1: List[int], shape2: List[int]
) -> Tuple[List[int], List[int]]:
    assert (
        len([d for d in list(shape1) + list(shape2) if d < 0]) <= 1
    ), "Only one dynamic dimension is allowed"
    shape1_dynamic_dims = [i for i, d in enumerate(shape1) if d <= 0]
    if len(shape1_dynamic_dims) > 0:
        s2, s1 = _reshape_infer_dynamic_dim(shape2, shape1)
        return s1, s2

    shape2_dynamic_dims = [i for i, d in enumerate(shape2) if d <= 0]
    if len(shape2_dynamic_dims) == 0:
        return shape1, shape2
    shape2_dynamic_dim = shape2_dynamic_dims[0]
    shape1_size = math.prod(shape1)
    shape2_size_without_dynamic_dim = math.prod(d for d in shape2 if d > 0)
    shape2_res = list(shape2)
    assert shape1_size % shape2_size_without_dynamic_dim == 0
    shape2_res[shape2_dynamic_dim] = shape1_size // shape2_size_without_dynamic_dim
    assert shape2_res[shape2_dynamic_dim] > 0
    return shape1, shape2_res


def _reshape_get_single_split_dim(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would split a single dimension, return its index and the length of the new dimensions.
    If the reshape is not of that kind return `None`.
    E.g.
    _reshape_get_single_split_dim(from_shape=(2, 12, 5), to_shape=(2, 3, 4, 5))
    results in
    (1, 2)"""
    from_shape, to_shape = _reshape_infer_dynamic_dim(from_shape, to_shape)

    if len(to_shape) < len(from_shape):
        return None
    i = longest_equal_range(from_shape, to_shape)
    split_dims_length = len(to_shape) - len(from_shape) + 1
    if i == len(from_shape):
        return (
            i,
            split_dims_length,
        )
    j = len(to_shape) - longest_equal_range(reversed(from_shape), reversed(to_shape))
    assert i < j
    expected_split_dim_size = math.prod(to_shape[i:j])
    if expected_split_dim_size == 1:
        # 1's were inserted.
        return (
            i,
            split_dims_length,
        )
    if expected_split_dim_size != from_shape[i]:
        return None
    return (
        i,
        split_dims_length,
    )


@unsqueeze.override(SplitPrimitiveTensor)
def unsqueeze_split(tensor: SplitPrimitiveTensor, dim: int) -> SplitPrimitiveTensor:
    shards = [torch.unsqueeze(unbox_tensor(shard), dim) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    dim_resolved = dim if dim >= 0 else dim + len(tensor.shape) + 1
    if shard_dim >= dim_resolved:
        shard_dim += 1
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


@unsqueeze.override(ReplicatedTensor)
def unsqueeze_replicated(tensor: ReplicatedTensor, dim: int) -> SplitPrimitiveTensor:
    shards = [torch.unsqueeze(unbox_tensor(shard), dim) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@view.override(ReplicatedTensor)
def view_replicated(tensor: ReplicatedTensor, shape: List[int]) -> ReplicatedTensor:
    view_split_range = _reshape_get_single_split_dim(tensor.shape, shape)
    if view_split_range is None:
        raise ValueError(
            "Only taking a tensor view where splitting a single dimension is supported"
        )
    shards = [view(shard, shape) for shard in tensor.shards]
    res = ReplicatedTensor(ts=shards)
    assert math.prod(res.shape) == math.prod(tensor.shape)
    return res


@view.override(SplitPrimitiveTensor)
def view_split(tensor: SplitPrimitiveTensor, shape: List[int]) -> SplitPrimitiveTensor:
    view_split_range = _reshape_get_single_split_dim(tensor.shape, shape)
    if view_split_range is None:
        raise ValueError(
            "Only taking a tensor view where splitting a single dimension is supported"
        )
    view_split_dim = view_split_range[0]

    if view_split_dim == tensor.shard_dim:
        if tensor.shape[view_split_dim] % tensor.shard_count != 0:
            raise ValueError(
                "Only splitting a dimension that is multiple of the shard count is supported"
            )
        if shape[view_split_dim] % tensor.shard_count != 0:
            raise ValueError(
                "The resulting leading splitting dimension must be multiple of the shard count"
            )

    shard_dim = tensor.shard_dim
    if shard_dim > view_split_dim:
        new_dims_count = len(shape) - len(tensor.shape)
        shard_dim += new_dims_count
    new_shard_shape = list(shape)
    new_shard_shape[shard_dim] //= tensor.shard_count
    shards = [view(shard, new_shard_shape) for shard in tensor.shards]
    res = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
    assert math.prod(res.shape) == math.prod(tensor.shape)
    return res


@view_as_complex.override(SplitPrimitiveTensor)
def view_as_complex_split(tensor: SplitPrimitiveTensor) -> SplitPrimitiveTensor:
    shards = [view_as_complex(shard) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@view_as_complex.override(ReplicatedTensor)
def view_as_complex_rep(tensor: ReplicatedTensor) -> ReplicatedTensor:
    shards = [view_as_complex(shard) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


@view_as_real.override(SplitPrimitiveTensor)
def view_as_real_split(tensor: SplitPrimitiveTensor) -> SplitPrimitiveTensor:
    shards = [view_as_real(shard) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@view_as_real.override(ReplicatedTensor)
def view_as_real_rep(tensor: ReplicatedTensor) -> ReplicatedTensor:
    shards = [view_as_real(shard) for shard in tensor.shards]
    return ReplicatedTensor(ts=shards)


# Note: Must be last thing in file
sharded_unwrap_override()
