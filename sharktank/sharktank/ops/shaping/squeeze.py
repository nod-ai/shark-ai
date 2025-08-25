# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import torch
from sharktank.types import AnyTensor, unbox_tensor
from sharktank.ops._registry import overridable


@overridable(dispatch_args=(0,))
def squeeze(tensor, dim: Optional[int]) -> AnyTensor:
    """See torch.squeeze"""
    ...


@squeeze.override(AnyTensor)
def squeeze_default(tensor, dim: Optional[int] = None) -> AnyTensor:
    if dim is None:
        return torch.squeeze(unbox_tensor(tensor))
    else:
        return torch.squeeze(unbox_tensor(tensor), dim)
