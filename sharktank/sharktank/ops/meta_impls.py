# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides for torch.fx.Proxy objects to enable FX tracing
# support for Brevitas quantization and other tracing scenarios.

from typing import Union
import torch
from torch import Tensor
from sharktank.types import PrimitiveTensor, QuantizedTensor, InferenceTensor
from sharktank.types.tensors import unbox_tensor
from .signatures import *

# Union type for meta implementations that includes Proxy objects
MetaAnyTensor = Union[torch.fx.Proxy, Tensor, InferenceTensor]

# During FX tracing with Proxy objects, we can't do control flow comparisons
# but we can still call operations - PyTorch will handle the dtype conversion
# automatically during actual execution
def linear_meta(input, weight, bias, *, accum_dtype) -> Tensor:
    """Linear implementation that handles torch.fx.Proxy objects during FX tracing."""
    input = unbox_tensor(input)
    weight = unbox_tensor(weight)
    bias = None if bias is None else unbox_tensor(bias)

    result = matmul(input, weight, transpose_rhs=True)
    if bias is not None:
        result = result + bias
    return result


# Linear overrides
linear.override(MetaAnyTensor, MetaAnyTensor, auto_dequant=True)(linear_meta)
linear.override(MetaAnyTensor, MetaAnyTensor, MetaAnyTensor, auto_dequant=True)(
    linear_meta
)


def matmul_meta(lhs, rhs, *, transpose_rhs: bool = False) -> Tensor:
    """Matmul implementation that handles torch.fx.Proxy objects during FX tracing."""
    lhs = unbox_tensor(lhs)
    rhs = unbox_tensor(rhs)

    if transpose_rhs:
        rhs = rhs.T

    return torch.matmul(lhs, rhs)


# Matmul overrides
matmul.override(MetaAnyTensor, MetaAnyTensor)(matmul_meta)
