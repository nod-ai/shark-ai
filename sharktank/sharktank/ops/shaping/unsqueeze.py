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


@overridable(dispatch_args=(0,))
def unsqueeze(tensor: AnyTensor, dim: int) -> AnyTensor:
    """See torch.unsqueeze"""
    ...


@unsqueeze.override(torch.Tensor)
def unsqueeze_default(
    tensor: Union[torch.Tensor, PrimitiveTensor], dim: int
) -> torch.Tensor:
    return torch.unsqueeze(unbox_tensor(tensor), dim)


@unsqueeze.override(QuantizedTensor)
def unsqueeze_quantized(tensor: QuantizedTensor, dim: int) -> QuantizedTensor:
    unpacked = tensor.unpack()
    if isinstance(unpacked, TensorScaledLayout):
        new_qs = unpacked._qs.unsqueeze(dim)
        layout = TensorScaledLayout(
            shape=new_qs.shape,
            d=unpacked._d,
            qs=new_qs,
            m=unpacked._m,
            dtype=unpacked.dtype,
        )
        return PlanarQuantizedTensor(shape=new_qs.shape, layout=layout)
    return NotImplemented


@unsqueeze.override(SplitPrimitiveTensor)
def unsqueeze_split(tensor: SplitPrimitiveTensor, dim: int) -> SplitPrimitiveTensor:
    shards = [torch.unsqueeze(unbox_tensor(shard), dim) for shard in tensor.shards]
    shard_dim = tensor.shard_dim
    dim_resolved = dim if dim >= 0 else dim + len(tensor.shape) + 1
    if shard_dim >= dim_resolved:
        shard_dim += 1
    return SplitPrimitiveTensor(ts=shards, shard_dim=shard_dim)
