# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List
import torch
from sharktank.types import (
    AnyTensor,
    unbox_tensor,
    QuantizedTensor,
    TensorScaledLayout,
    PlanarQuantizedTensor,
    SplitPrimitiveTensor,
)
from sharktank.ops._registry import overridable


@overridable(dispatch_args=(0,))
def expand(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    """See torch.Tensor.expand"""
    ...


@expand.override(torch.Tensor)
def expand_default(tensor: AnyTensor, shape: List[int]) -> AnyTensor:
    return unbox_tensor(tensor).expand(*shape)


@expand.override(QuantizedTensor)
def expand_quantized(tensor: QuantizedTensor, shape: List[int]) -> QuantizedTensor:
    unpacked = tensor.unpack()
    if isinstance(unpacked, TensorScaledLayout):
        new_qs = unpacked._qs.expand(*shape)
        layout = TensorScaledLayout(
            shape=new_qs.shape,
            d=unpacked._d,
            qs=new_qs,
            m=unpacked._m,
            dtype=unpacked.dtype,
        )
        return PlanarQuantizedTensor(shape=new_qs.shape, layout=layout)
    return NotImplemented


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
