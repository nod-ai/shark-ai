# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from sharktank.types.tensors import (
    InferenceTensor,
    PrimitiveTensor,
    DefaultPrimitiveTensor,
)
from sharktank import ops


def set_float_dtype(tensor: InferenceTensor, dtype: torch.dtype) -> InferenceTensor:
    if isinstance(tensor, PrimitiveTensor) and tensor.dtype.is_floating_point:
        return DefaultPrimitiveTensor(
            name=tensor.name, data=ops.to(tensor, dtype=dtype)
        )

    return tensor
