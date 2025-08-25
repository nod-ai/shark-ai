# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List
import torch
from torch import Tensor
from sharktank.types import (
    AnyTensor,
    unbox_tensor,
    SplitPrimitiveTensor,
)
from sharktank.ops._registry import overridable


@overridable(dispatch_args=(0,))
def permute(tensor: AnyTensor, dims: List[int]) -> AnyTensor:
    """Permute the tensor dimensions according to the permutation `dims` in line
    notation.
    The semantics are the same as torch.permute."""
    ...


@permute.override(Tensor)
def permute_default(tensor: Tensor, dims: List[int]):
    torch_tensor = unbox_tensor(tensor)
    return torch.permute(torch_tensor, dims)


@permute.override(SplitPrimitiveTensor)
def permute_split(tensor: SplitPrimitiveTensor, dims: List[int]):
    permuted_shards = [permute(shard, dims) for shard in tensor.shards]
    permuted_shard_dim = dims[tensor.shard_dim]
    return SplitPrimitiveTensor(ts=permuted_shards, shard_dim=permuted_shard_dim)
