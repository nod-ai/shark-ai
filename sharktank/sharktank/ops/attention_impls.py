# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

from typing import Optional

import math
import torch
import warnings


from torch import Tensor

from sharktank import kernels

from types import NoneType

from ..types import (
    AnyTensor,
    PlanarQuantizedTensor,
)
from ..kernels import flash_attention, masked_flash_attention

from ..types.layouts import TensorScaledLayout

from ..utils import debugging

from ..types.tensors import unbox_tensor
from .signatures import (
    IntOrSequenceInt,
    scaled_dot_product_attention,
    elementwise,
)


def _extract_linear_scale(t):
    if (
        isinstance(t, PlanarQuantizedTensor)
        and isinstance(t.layout, TensorScaledLayout)
        and t.layout.m is None
    ):
        return t.layout.qs, t.layout.d
    return unbox_tensor(t), None


def register_attention_override_by_name(name: str):
    """Provides a way to override available attention kernels
    based on something other than a global flag"""
    if name == "flash_attention":
        scaled_dot_product_attention.override(
            PlanarQuantizedTensor,
            PlanarQuantizedTensor,
            PlanarQuantizedTensor,
            NoneType,
        )(flash_attention)
    elif name == "masked_flash_attention":
        scaled_dot_product_attention.override(
            AnyTensor, AnyTensor, AnyTensor, AnyTensor
        )(masked_flash_attention)
    else:
        assert False, f"{name} not a registerable override"


def prepare_args(q, k, v, scale):
    scale = torch.scalar_tensor(1.0 / math.sqrt(q.shape[-1]), dtype=torch.float32)

    q, qscale = _extract_linear_scale(q)
    k, kscale = _extract_linear_scale(k)
    v, vscale = _extract_linear_scale(v)

    scale = scale * qscale if qscale is not None else scale
    scale = scale * kscale if kscale is not None else scale

    if q.dtype == torch.float32:
        q = q.to(torch.float16)

    if k.dtype == torch.float32:
        k = k.to(torch.float16)

    if v.dtype == torch.float32:
        v = v.to(torch.float16)

    atten = kernels.flash_attention(q, k, v, a, scale)

    atten = atten * vscale if vscale is not None else atten
    return atten


if debugging.flags.use_custom_iree_kernels:
    scaled_dot_product_attention.override(
        PlanarQuantizedTensor, PlanarQuantizedTensor, PlanarQuantizedTensor, NoneType
    )(flash_attention)
