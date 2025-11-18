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

from amdsharktank.types import (
    AnyTensor,
    BlockScaledFp4Layout,
    BlockScaledLayout,
    DefaultPrimitiveTensor,
    InferenceTensor,
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
    is_any_tensor,
    unbox_tensor,
)
from ._registry import (
    AllOfExprs,
    AllOfType,
    AllOfExprsVariadic,
    AnyOfType,
    BoolTypeExpr,
    IsOfType,
    SignatureDispatcher,
)
from .shape import (
    broadcast_dims,
    broadcast_dim,
    unbroadcast_dim,
    normalize_negative_dim,
)
from amdsharktank.utils import longest_equal_range, tree
from amdsharktank.utils.math import ceildiv
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
        Wrapper for each NON-TRANSFERRING op defined in this file.
        """

        def func_wrapper(*args: Tuple, **kwargs: Dict[str, Any]):
            """
            Wraps each NON-TRANSFERRING operation, f, to ensure that all incoming tensors are on the same device and that the result has the devices correctly labelled.

            If no ShardedTensors are present in the input, then no changes are made to input/output.
            """
            sharded_tensors = []
            for value in itertools.chain(args, kwargs.values()):
                if isinstance(value, ShardedTensor):
                    sharded_tensors.append(value)
                    continue
                if isinstance(
                    value,
                    (
                        InferenceTensor,
                        Tensor,
                    ),
                ):
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

        func_wrapper._impl_name = getattr(f, "_impl_name", None)  # For impl selection
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


def _register_trivially_replicable():
    from . import signatures
    from .utils import trivially_replicable

    def replicated_if_tensor(t: type) -> bool:
        if issubclass(t, ReplicatedTensor):
            return True
        if not issubclass(t, (Tensor, InferenceTensor)):
            return True
        return False

    def should_override(*types: tuple[type]) -> bool:
        at_least_one_replicated_tensor = any(
            issubclass(t, ReplicatedTensor) for t in types
        )
        if not at_least_one_replicated_tensor:
            return False
        return all(replicated_if_tensor(t) for t in types)

    for func_name in signatures.__all__:
        func = globals()[func_name]
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


@arange.override()
def arange_replicated(
    *start_end_step,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
):
    if devices is None:
        return NotImplemented

    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the arange result from being fused.
    shards = [arange(*start_end_step, dtype=dtype, device=device) for _ in devices]
    return ReplicatedTensor(ts=shards, devices=tuple(devices))


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


@full.override()
def full_replicated(
    size: Sequence[int],
    fill_value: Number,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
):
    if devices is None:
        return NotImplemented

    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the results from being fused.
    shards = [full(size, fill_value, dtype=dtype, device=device) for _ in devices]
    return ReplicatedTensor(ts=shards, devices=tuple(devices))


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
    input: ShardedTensor,
    weight: ShardedTensor,
    bias: ShardedTensor | None,
    *,
    accum_dtype,
    matmul_impl=None,
) -> SplitPrimitiveTensor:
    if (
        isinstance(input, ReplicatedTensor)
        and isinstance(weight, ReplicatedTensor)
        and (bias is None or isinstance(bias, ReplicatedTensor))
    ):
        # Delegate to the standard handling of the trivially replicable mechanism when
        # all are replicated.
        return NotImplemented

    # TODO: handle different dtypes
    result = matmul(input, weight, transpose_rhs=True, impl=matmul_impl)
    if bias is not None:
        result = elementwise(torch.add, result, bias)
    return result


linear.override(ShardedTensor, ShardedTensor)(linear_sharded)
linear.override(ShardedTensor, ShardedTensor, ShardedTensor)(linear_sharded)


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


@ones.override()
def ones_replicated(
    *size,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
):
    if devices is None:
        return NotImplemented

    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the results from being fused.
    shards = [ones(*size, dtype=dtype, device=device) for _ in devices]
    return ReplicatedTensor(ts=shards, devices=tuple(devices))


@ones_like.override(AllOfType(ReplicatedTensor, SplitPrimitiveTensor))
def ones_like_replicated(
    tensor: ReplicatedTensor | SplitPrimitiveTensor,
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> ReplicatedTensor | SplitPrimitiveTensor:
    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the results from being fused.
    shards = [ones_like(shard, dtype=dtype, device=device) for shard in tensor.shards]
    return tensor.clone(ts=shards)


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


@permute.override(SplitPrimitiveTensor)
def permute_split(tensor: SplitPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return SplitPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)


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
        elif isinstance(result, Tensor):
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


def tensor_replicated(
    data: AnyTensor | Number,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    devices: Sequence[int] | None = None,
) -> ReplicatedTensor:
    if devices is None and not isinstance(data, ReplicatedTensor):
        return NotImplemented
    if isinstance(data, AnyTensor) and not isinstance(data, ReplicatedTensor):
        return NotImplemented  # Don't want to implicitly create transfers
    if devices is not None and isinstance(data, ReplicatedTensor):
        assert tuple(devices) == data.devices

    if isinstance(data, ReplicatedTensor):
        data_shards = data.shards
        devices = data.devices
    else:
        data_shards = [data] * len(devices)

    shards = [tensor(shard, dtype=dtype, device=device) for shard in data_shards]
    return ReplicatedTensor(ts=shards, devices=devices)


tensor.override()(tensor_replicated)
tensor.override(ReplicatedTensor)(tensor_replicated)


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


@transpose.override(SplitPrimitiveTensor)
def transpose_split(
    tensor: SplitPrimitiveTensor, dim0: int, dim1: int
) -> SplitPrimitiveTensor:
    shards = [transpose(shard, dim0, dim1) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    if dim0 < 0:
        dim0 = len(tensor.shape) + dim0
    if dim1 < 0:
        dim1 = len(tensor.shape) + dim1
    if shard_dim == dim0:
        shard_dim = dim1
    elif shard_dim == dim1:
        shard_dim = dim0
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)


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


@unpack.override(IsOfType(ReplicatedTensor, SplitPrimitiveTensor))
def unpack_sharded(input: ReplicatedTensor | SplitPrimitiveTensor) -> QuantizedLayout:
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
        if isinstance(input, ReplicatedTensor) or len(shards[0].shape) == 0:
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


@view.override(SplitPrimitiveTensor)
def view_split(
    tensor: SplitPrimitiveTensor, shape: List[int] | None, dtype: torch.dtype | None
) -> SplitPrimitiveTensor:
    assert dtype is None, "Not supported"
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


@view_as_real.override(SplitPrimitiveTensor)
def view_as_real_split(tensor: SplitPrimitiveTensor) -> SplitPrimitiveTensor:
    shards = [view_as_real(shard) for shard in tensor.shards]
    return SplitPrimitiveTensor(ts=shards, shard_dim=tensor.shard_dim)


@zeros.override()
def zeros_replicated(
    *size,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    devices: Sequence[int] | None = None,
):
    if devices is None:
        return NotImplemented

    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the results from being fused.
    shards = [zeros(*size, dtype=dtype, device=device) for _ in devices]
    return ReplicatedTensor(ts=shards, devices=tuple(devices))


@zeros_like.override(AllOfType(ReplicatedTensor, SplitPrimitiveTensor))
def zeros_like_replicated(
    tensor: ReplicatedTensor | SplitPrimitiveTensor,
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> ReplicatedTensor | SplitPrimitiveTensor:
    # Do not use `tranfer_to_logical_device` here.
    # Adding the transfer op would prevent the results from being fused.
    shards = [zeros_like(shard, dtype=dtype, device=device) for shard in tensor.shards]
    return tensor.clone(ts=shards)


# Note: Must be last thing in file
sharded_unwrap_override()
