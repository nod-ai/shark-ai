# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

from types import NoneType
import math
import torch

from sharktank import kernels
from sharktank.types import (
    AnyTensor,
    PlanarQuantizedTensor,
)
from sharktank import kernels

from sharktank.types.layouts import TensorScaledLayout

from sharktank.utils import debugging

from sharktank.types.tensors import ReplicatedTensor, unbox_tensor
from .signatures import (
    scaled_dot_product_attention,
)


def _extract_linear_scale(t):
    if (
        isinstance(t, PlanarQuantizedTensor)
        and isinstance(t.layout, TensorScaledLayout)
        and t.layout.m is None
    ):
        return t.layout.qs, t.layout.d
    return unbox_tensor(t), None


@scaled_dot_product_attention.override(
    PlanarQuantizedTensor,
    PlanarQuantizedTensor,
    PlanarQuantizedTensor,
    torch.Tensor,
)
def masked_flash_attention(q, k, v, a, is_causal, scale):
    if is_causal:
        return NotImplemented

    if scale is None:
        scale = torch.scalar_tensor(1.0 / math.sqrt(q.shape[-1]), dtype=torch.float32)

    q, qscale = _extract_linear_scale(q)
    k, kscale = _extract_linear_scale(k)
    v, vscale = _extract_linear_scale(v)

    scale = scale * qscale if qscale is not None else scale
    scale = scale * kscale if kscale is not None else scale

    atten = kernels.masked_flash_attention(q, k, v, a, scale)

    atten = atten * vscale if vscale is not None else atten
    return atten


@scaled_dot_product_attention.override(
    PlanarQuantizedTensor,
    PlanarQuantizedTensor,
    PlanarQuantizedTensor,
    NoneType,
)
def flash_attention(q, k, v, is_causal, scale):
    if is_causal:
        return NotImplemented

    if scale is None:
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

    atten = kernels.flash_attention(q, k, v, scale)

    atten = atten * vscale if vscale is not None else atten
    return atten
