# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union
import torch
from sharktank.types import (
    AnyTensor,
    PrimitiveTensor,
    unbox_tensor,
    QuantizedTensor,
    TensorScaledLayout,
    PlanarQuantizedTensor,
    SplitPrimitiveTensor,
)
from sharktank.ops._registry import overridable
from sharktank.ops.quantized_impls import quantized_tensor_layout_of_type


@overridable(dispatch_args=(0,))
def flatten(input: AnyTensor, start_dim: int = 0, end_dim: int = -1) -> AnyTensor:
    """See torch.flatten"""
    ...


@flatten.override(torch.Tensor)
def flatten_default(
    input: Union[torch.Tensor, PrimitiveTensor], start_dim: int, end_dim: int
) -> torch.Tensor:
    return torch.flatten(unbox_tensor(input), start_dim, end_dim)


@flatten.override(QuantizedTensor)
@quantized_tensor_layout_of_type(tensor=TensorScaledLayout)
def flatten_tensor_scaled_layout(
    tensor: QuantizedTensor, start_dim: int, end_dim: int
) -> QuantizedTensor:
    unpacked = tensor.unpack()
    new_qs = torch.flatten(unpacked._qs, start_dim, end_dim)
    layout = TensorScaledLayout(
        shape=new_qs.shape,
        d=unpacked._d,
        qs=new_qs,
        m=unpacked._m,
        dtype=unpacked.dtype,
    )
    return PlanarQuantizedTensor(shape=new_qs.shape, layout=layout)


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
