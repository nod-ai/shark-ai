# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from ._registry import AllOfType
from .signatures import *
from amdsharktank.types import (
    DefaultPrimitiveTensor,
    PrimitiveTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
)
from torch import Tensor
from typing import Sequence


@cat.override(AllOfType(Tensor, PrimitiveTensor))
def cat_cpu_eager_f8(
    tensors: Sequence[Tensor | PrimitiveTensor], dim: int
) -> Tensor | PrimitiveTensor:
    """A workaround for torch not having CPU implementation for f8.
    During export we don't want to bitcast to int8 as the IREE compiler can't optimize
    out the bitcast in some cases."""
    is_inference_tensor = isinstance(tensors[0], PrimitiveTensor)
    tensors = [unbox_tensor(t) for t in tensors]

    int8_like_dtypes = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    if torch.compiler.is_compiling() or not all(
        t.dtype in int8_like_dtypes and t.is_cpu for t in tensors
    ):
        return NotImplemented

    orig_dtype = tensors[0].dtype
    tensors = [t.view(dtype=torch.int8) for t in tensors]
    result = cat(tensors, dim)
    result = result.view(dtype=orig_dtype)

    if is_inference_tensor:
        result = DefaultPrimitiveTensor(data=result)
    return result
