# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from typing import Sequence
from amdsharktank.types.tensors import (
    InferenceTensor,
    PrimitiveTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
)
from amdsharktank import ops


def set_float_dtype(tensor: InferenceTensor, dtype: torch.dtype) -> InferenceTensor:
    if isinstance(tensor, PrimitiveTensor) and tensor.dtype.is_floating_point:
        return DefaultPrimitiveTensor(
            name=tensor.name, data=unbox_tensor(ops.to(tensor, dtype=dtype))
        )

    return tensor


def convert_dtype(
    tensor: InferenceTensor, dtype_map: dict[torch.dtype, torch.dtype]
) -> InferenceTensor:
    for src_dtype, tgt_dtype in dtype_map.items():
        if src_dtype == tensor.dtype:
            res = tensor.to(dtype=tgt_dtype)
            res.name = tensor.name
            return res

    return tensor


def wrap_in_list_if_inference_tensor(
    arg: InferenceTensor | Sequence[InferenceTensor] | None,
) -> list[InferenceTensor] | None:
    if arg is None:
        return None
    if isinstance(arg, InferenceTensor):
        return [arg]
    return arg
