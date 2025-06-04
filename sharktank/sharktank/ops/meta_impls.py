# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains overrides for torch.fx.Proxy objects to enable FX tracing
# support for Brevitas quantization and other tracing scenarios.

from typing import Union
import torch
from torch import Tensor
from sharktank.types import InferenceTensor
from sharktank.types.tensors import unbox_tensor
from .signatures import *

# Union type for meta implementations that includes Proxy objects
MetaAnyTensor = Union[torch.fx.Proxy, Tensor, InferenceTensor]


def any_input_meta(*args):
    for arg in args:
        if isinstance(arg, torch.fx.Proxy):
            return True
    return False


def meta_unbox_tensor(tensor):
    if isinstance(tensor, torch.fx.Proxy):
        return tensor
    else:
        return unbox_tensor(tensor)


# During FX tracing with Proxy objects, we can't do control flow comparisons
# but we can still call operations - PyTorch will handle the dtype conversion
# automatically during actual execution
def linear_meta(input, weight, bias, *, accum_dtype) -> Tensor:
    if not any_input_meta(input, weight, bias):
        return NotImplemented
    input = meta_unbox_tensor(input)
    weight = meta_unbox_tensor(weight)
    bias = None if bias is None else unbox_tensor(bias)

    result = matmul(input, weight, transpose_rhs=True)
    if bias is not None:
        result = result + bias
    return result


linear.override(MetaAnyTensor, MetaAnyTensor, auto_dequant=True)(linear_meta)
linear.override(MetaAnyTensor, MetaAnyTensor, MetaAnyTensor, auto_dequant=True)(
    linear_meta
)


def matmul_meta(lhs, rhs, *, transpose_rhs: bool = False) -> Tensor:
    if not any_input_meta(lhs, rhs):
        return NotImplemented
    lhs = meta_unbox_tensor(lhs)
    rhs = meta_unbox_tensor(rhs)

    if transpose_rhs:
        rhs = rhs.T

    return torch.matmul(lhs, rhs)


matmul.override(MetaAnyTensor, MetaAnyTensor)(matmul_meta)
