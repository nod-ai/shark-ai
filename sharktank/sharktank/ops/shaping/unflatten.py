# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union, Tuple
import torch
from torch import Tensor
from sharktank.types import (
    AnyTensor,
    PrimitiveTensor,
    unbox_tensor,
    SplitPrimitiveTensor,
)
from sharktank.ops._registry import overridable


@overridable(dispatch_args=(0,))
def unflatten(input: AnyTensor, dim: int, sizes: Tuple[int]) -> AnyTensor:
    """See torch.unflatten"""
    ...


@unflatten.override(Tensor)
def unflatten_default(
    input: Union[Tensor, PrimitiveTensor], dim: int, sizes: Tuple[int]
) -> Tensor:
    return torch.unflatten(unbox_tensor(input), dim, sizes)


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
