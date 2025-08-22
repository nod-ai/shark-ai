# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Union, Optional, Tuple
import math
import torch
from torch import Tensor
from sharktank.types import (
    AnyTensor,
    PrimitiveTensor,
    unbox_tensor,
    ReplicatedTensor,
    SplitPrimitiveTensor,
)
from sharktank.ops._registry import overridable
from sharktank.ops.sharding.utils import (
    _reshape_infer_dynamic_dim,
    _reshape_get_single_split_dim,
    _reshape_get_flatten_dim_range,
)
from .view import view
from .flatten import flatten


@overridable(dispatch_args=(0,))
def reshape(input: AnyTensor, shape: List[int]) -> AnyTensor:
    """Returns a tensor with the same data and number of elements as input, but with
    the specified shape.
    See torch.reshape.
    """
    ...


@reshape.override(Tensor)
def reshape_default(input: Union[PrimitiveTensor, Tensor], shape: List[int]) -> Tensor:
    return torch.reshape(unbox_tensor(input), shape)


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
