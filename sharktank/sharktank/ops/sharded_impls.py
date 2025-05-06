# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import List, Optional, Sequence, Union, Any, Tuple, Dict, Iterable
import itertools
from numbers import Number
import math
import functools
import torch
from torch import Tensor

from sharktank.types import (
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
from sharktank.types.tensors import unbox_tensor
from ._registry import AllOfType, AllOfExprsVariadic, AnyOfType, IsOfType
from .shape import broadcast_dims, broadcast_dim, unbroadcast_dim
from sharktank.utils import longest_equal_range
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
            if len(sharded_tensors) > 0:
                if isinstance(res, ShardedTensor):
                    res = res.clone(devices=sharded_tensors[0].devices)
                elif isinstance(res, Iterable) and all(
                    isinstance(r, ShardedTensor) for r in res
                ):
                    res = type(res)(
                        r.clone(devices=sharded_tensors[0].devices) for r in res
                    )
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
        shard_dim=x_shard_dim,
        shape=torch.broadcast_shapes(x.shape, y.shape),
        ts=partials,
    )


@einsum_2args.override(AllOfType(ReplicatedTensor))
def einsum_2args_replicated(
    input0: ReplicatedTensor, input1: ReplicatedTensor, einsum_str: str
) -> ReplicatedTensor:
    assert input0.shard_count == input1.shard_count
    shards = [
        einsum_2args(input0_shard, input1_shard, einsum_str)
        for input0_shard, input1_shard in zip(input0.shards, input1.shards)
    ]
    return ReplicatedTensor(ts=shards)


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


@index_put_.override(AllOfType(ReplicatedTensor))
def index_put__replicated(
    inout: ReplicatedTensor,
    indices: Tuple[ReplicatedTensor],
    values: ReplicatedTensor,
) -> ReplicatedTensor:
    assert inout.shard_count == values.shard_count
    for i, shard in enumerate(inout.shards):
        shard_indices = [idx.shards[i] for idx in indices]
        shard.index_put_(shard_indices, values.shards[i])
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


@masked_fill.override(AllOfType(ReplicatedTensor))
def masked_fill_replicated(
    tensor: ReplicatedTensor,
    mask: ReplicatedTensor,
    value: Number,
) -> ReplicatedTensor:
    assert tensor.shard_count == mask.shard_count
    shards = [
        shard.masked_fill(mask_shard, value)
        for shard, mask_shard in zip(tensor.shards, mask.shards)
    ]
    return ReplicatedTensor(ts=shards)


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
) -> ReplicatedTensor:
    shards = [mean(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in x.shards]
    return ReplicatedTensor(ts=shards)


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
        gathered = cat(
            [
                (
                    transfer_to_logical_device(shard, x.devices[0])
                    if i != 0
                    else barrier_on_logical_device(shard, x.devices[0])
                )
                for i, shard in enumerate(x.shards)
            ],
            dim=x.shard_dim,
        )
        meaned = mean(gathered, dim=dim, keepdim=keepdim, dtype=dtype)
        return ReplicatedTensor(ts=meaned, shard_count=x.shard_count, devices=x.devices)


@module_register_buffer.override(torch.nn.Module, ShardedTensor)
def module_register_buffer_sharded(
    module: torch.nn.Module, name: str, tensor: ShardedTensor
) -> None:
    for i, shard in enumerate(tensor.shards):
        module_register_buffer(module, f"{name}__shard__{i}", shard)
    setattr(module, name, tensor)


@pad.override(ReplicatedTensor)
def pad_replicated(
    input: ReplicatedTensor,
    _pad: List[int],
    mode: str = None,
    value: Optional[float] = None,
) -> ReplicatedTensor:
    shards = [pad(shard, _pad=_pad, mode=mode, value=value) for shard in input.shards]
    return ReplicatedTensor(ts=shards)


@pad.override(SplitPrimitiveTensor)
def pad_replicated(
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


@reshape.override(ReplicatedTensor)
def reshape_replicated(tensor: ReplicatedTensor, shape: List[int]) -> ReplicatedTensor:
    return ReplicatedTensor(ts=[reshape(shard, shape) for shard in tensor.shards])


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


@scatter_.override(ReplicatedTensor, ReplicatedTensor)
def scatter_replicated_replicated(
    inout: ReplicatedTensor,
    dim: int,
    index: ReplicatedTensor,
    value: Number,
    *,
    reduce: str = None,
) -> ReplicatedTensor:
    assert isinstance(value, Number), "Tensor version of this op not implemented"
    assert inout.shard_count == index.shard_count
    for shard, index_shard in zip(inout.shards, index.shards):
        scatter_(shard, dim, index_shard, value, reduce=reduce)
    return inout


@scatter_.override(SplitPrimitiveTensor, SplitPrimitiveTensor)
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


@sigmoid.override(ShardedTensor)
def sigmoid_sharded(tensor: ShardedTensor) -> ShardedTensor:
    return elementwise(torch.sigmoid, tensor)


@softmax.override(ReplicatedTensor)
def softmax_replicated(
    tensor: ReplicatedTensor, dim: Optional[int], dtype: Optional[torch.dtype]
) -> ReplicatedTensor:
    shards = [softmax(shard, dim=dim, dtype=dtype) for shard in tensor.shards]
    return tensor.clone(ts=shards)


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


@sum.override(ReplicatedTensor)
def sum_replicated(
    input: ReplicatedTensor,
    dim: int | List[int] | None,
    keepdim: bool,
    *,
    dtype: torch.dtype,
) -> ReplicatedTensor:
    assert dim is not None, "sum dim must be specified"
    shards = [
        sum(shard, dim=dim, keepdim=keepdim, dtype=dtype) for shard in input.shards
    ]
    return ReplicatedTensor(ts=shards)


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


@topk.override(ReplicatedTensor)
def topk_replicated(
    input: ReplicatedTensor, k: int, dim: int, largest: bool, sorted: bool
) -> tuple[ReplicatedTensor, ReplicatedTensor]:
    values, indices = zip(
        *(
            topk(shard, k=k, dim=dim, largest=largest, sorted=sorted)
            for shard in input.shards
        )
    )
    return ReplicatedTensor(ts=values), ReplicatedTensor(ts=indices)


@topk.override(SplitPrimitiveTensor)
def topk_split(
    input: SplitPrimitiveTensor, k: int, dim: int, largest: bool, sorted: bool
) -> tuple[
    SplitPrimitiveTensor | ReplicatedTensor, SplitPrimitiveTensor | ReplicatedTensor
]:
    if dim != input.shard_dim:
        values, indices = zip(
            *(
                topk(shard, k=k, dim=dim, largest=largest, sorted=sorted)
                for shard in input.shards
            )
        )
        values_split = SplitPrimitiveTensor(ts=values, shard_dim=input.shard_dim)
        indices_split = SplitPrimitiveTensor(ts=indices, shard_dim=input.shard_dim)
        return values_split, indices_split
    else:
        # TODO: Implement more efficient version which does a
        #       topk on each shard before transferring and then adjusts
        #       the indices.
        gathered = sharded_cat(input)
        values, indices = topk(gathered, k=k, dim=dim, largest=largest, sorted=sorted)
        values = replicate(values, count=input.shard_count, devices=input.devices)
        indices = replicate(indices, count=input.shard_count, devices=input.devices)
        return values, indices


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


def _calculate_view_dimension_mapping(
    from_shape: Sequence[int], to_shape: Sequence[int]
) -> List[List[int]]:
    """
    Calculate a mapping from the dimensions in `from_shape` to those in `to_shape`.
    """
    from_shape, to_shape = list(from_shape), list(to_shape)
    assert len(from_shape) > 0 and len(to_shape) > 0, "Scalars not supported"
    assert all(d != 0 for d in from_shape + to_shape), "Zero dimensions not supported"
    from_shape, to_shape = _reshape_infer_dynamic_dim(list(from_shape), list(to_shape))

    # Trivial cases
    if len(from_shape) == 1:
        return [[i for i in range(len(to_shape))]]
    if len(to_shape) == 1:
        return [[0] for _ in range(len(from_shape))]

    def _get_cumulative_boundaries(shape: Sequence[int]) -> List[int]:
        """
        Get the cumulitive number of elements at the start of each dimension.
        Add an extra 1 at the start to represent the start of the first dimension.
        For example, for shape (2, 3, 4) it returns [1, 2, 6, 24].
        """
        return [1] + list(itertools.accumulate(shape, lambda x, y: x * y))

    bounds_to = _get_cumulative_boundaries(to_shape)
    bounds_from = _get_cumulative_boundaries(from_shape)

    mapping = [[] for _ in range(len(from_shape))]
    to_dim_idx_start = 0
    for from_dim in range(len(from_shape)):
        from_bound_start = bounds_from[from_dim]
        from_bound_end = bounds_from[from_dim + 1]

        to_dim = to_dim_idx_start
        while to_dim < len(to_shape):
            to_bound_start = bounds_to[to_dim]
            to_bound_end = bounds_to[to_dim + 1]

            # Check if the two ranges overlap
            overlap_start = max(to_bound_start, from_bound_start)
            overlap_end = min(to_bound_end, from_bound_end)
            range_overlaps = overlap_start < overlap_end

            # Special case for dim of size 1
            size_one_dim_overlap = False
            if from_bound_start == from_bound_end:  # `from_dim` is 1
                if (
                    from_bound_start >= to_bound_start
                    and from_bound_start < to_bound_end
                ):
                    # `from_dim` is within the range of `to_dim`.
                    # E.g. [5, 1, 6] to [5, 6]
                    size_one_dim_overlap = True
                elif (
                    from_bound_start == to_bound_start
                    and from_bound_end == to_bound_start
                ):
                    size_one_dim_overlap = True

            if range_overlaps or size_one_dim_overlap:
                # Overlap exists
                assert to_dim not in mapping[from_dim]
                mapping[from_dim].append(to_dim)

                if to_bound_end >= from_bound_end:
                    # We have exhausted the current `from_dim`
                    if to_bound_end == from_bound_end:
                        # This `to_dim` ends *exactly* at the end of the current `from_dim`.
                        # This `to_dim` is exhausted, start next search with next `to_dim`.
                        to_dim_idx_start = to_dim + 1
                    else:  # to_bound_end > from_bound_end
                        # This `to_dim` ends *after* the current `from_dim` ends.
                        # We need to check the next `from_dim` for the current `to_dim`;
                        # This `to_dim` is split across multiple `from_dim`s.
                        to_dim_idx_start = to_dim
                    # Found all contributions of this `from_dim`, more to the next.
                    break
                else:  # to_bounds_end < from_bounds_end
                    # This to_dim ends *before* the current `from_dim` ends.
                    # We need to check the next to_dim for the current `from_dim`.
                    to_dim += 1
            elif to_bound_start > from_bound_end:
                # This `to_dim` starts *after* the current `from_dim` ends.
                # No further `to_dim`s will overlap this `from_dim`.
                # The next search should start from this `to_dim`.
                to_dim_idx_start = to_dim
                break
            else:  # to_bounds_end <= from_bounds_start
                # This `to_dim` ends *before* or *at* the start of the current `from_dim`.
                # Move to check the next `to_dim` for the current `from_dim`.
                to_dim += 1
        # Update search start if inner loop finishes by exhaustion
        if to_dim == len(to_shape):
            to_dim_idx_start = to_dim

        # Handle empty mapping for size 1 dimensions that didn't get mapped (happens if this is trailing 1)
        if from_shape[from_dim] == 1 and not mapping[from_dim]:
            last_valid_idx = len(to_shape) - 1
            mapping[from_dim].append(last_valid_idx)

    return mapping


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
        assert math.prod(shape1) == math.prod(
            shape2
        ), f"Size mismatch: {shape1} vs {shape2}"
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
    shards = [view(shard, shape) for shard in tensor.shards]
    res = ReplicatedTensor(ts=shards)
    assert math.prod(res.shape) == math.prod(tensor.shape)
    return res


@view.override(SplitPrimitiveTensor)
def view_split(tensor: SplitPrimitiveTensor, shape: List[int]) -> SplitPrimitiveTensor:
    shard_dim = tensor.shard_dim
    mapping = _calculate_view_dimension_mapping(from_shape=tensor.shape, to_shape=shape)
    if len(mapping[shard_dim]) != 1:
        if tensor.shape[tensor.shard_dim] % tensor.shard_count != 0:
            raise ValueError(
                "Only splitting a dimension that is multiple of the shard count is supported"
            )
        if shape[tensor.shard_dim] % tensor.shard_count != 0:
            raise ValueError(
                "The resulting leading splitting dimension must be multiple of the shard count"
            )

    # Account for collapsed or expanded dims
    collapsed_dims = []
    delta = 0
    for from_dim, to_dims in enumerate(mapping[: shard_dim + 1]):
        if len(to_dims) > 1:
            # Expanded dims move shard_dim to the right by 1 for each new dim.
            if from_dim == shard_dim:
                pass  # Do nothing since we want to shard based on the leading dim if the shard_dim is expanded.
            else:
                delta += len(to_dims) - 1
        # A to_dim can be split to be both expand itself and be collapsed with others, must check.
        for to_dim in to_dims:
            # Collapsed dims move shard_dim to the left by 1 for each dim after the first.
            if to_dim in collapsed_dims:
                delta -= 1
            collapsed_dims.append(to_dim)
    # Account for extra dims of size 1
    dims_not_seen = [i for i in range(min(mapping[shard_dim]))]
    for to_dims in mapping[:shard_dim]:
        for to_dim in to_dims:
            if to_dim in dims_not_seen:
                dims_not_seen.remove(to_dim)

    shard_dim += delta + len(dims_not_seen)

    new_shard_shape = list(shape)
    # NOTE: dynamic shard_dim is handled implicitly because of int division.
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
