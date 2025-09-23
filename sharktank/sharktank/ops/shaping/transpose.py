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
    DefaultPrimitiveTensor,
    unbox_tensor,
    SplitPrimitiveTensor,
    PlanarQuantizedTensor,
    BlockScaledLayout,
)
from sharktank.ops._registry import overridable
from ..sharding.utils import wrap_override


@overridable(dispatch_args=(0,))
def transpose(tensor: AnyTensor, dim0: int, dim1: int) -> AnyTensor:
    """See torch.transpose"""
    ...


@transpose.override(torch.Tensor)
def transpose_default(
    tensor: Union[torch.Tensor, PrimitiveTensor], dim0: int, dim1: int
) -> Union[torch.Tensor, PrimitiveTensor]:
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


@wrap_override(transpose.override)(SplitPrimitiveTensor)
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
