# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

import math
import torch

from sharktank import kernels, ops
from sharktank.types import (
    AnyTensor,
    PlanarQuantizedTensor,
)

from sharktank.types.layouts import TensorScaledLayout

from sharktank.types.tensors import unbox_tensor
from .signatures import (
    scaled_dot_product_attention,
)
from ._registry import AnyType

# These two versions should be preserved in this order
@scaled_dot_product_attention.override(
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyType,
    impl_name="decomposed",
)
def scaled_dot_product_attention_decomposed(
    q, k, v, a, is_causal, scale, softcap, impl
):

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    attn_weights = torch.matmul(
        q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)
    )
    attn_weights = attn_weights * scale

    if softcap is not None:
        attn_weights = softcap * torch.tanh(attn_weights / softcap)

    if a is not None:
        attn_weights = attn_weights + a
    elif is_causal:
        mask = torch.full((attn_weights.shape[2], attn_weights.shape[3]), float("-inf"))
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        attn_weights = attn_weights + mask

    attn_weights = ops.softmax(ops.to(attn_weights, dtype=torch.float32), dim=-1)
    attn_weights = unbox_tensor(ops.to(attn_weights, dtype=q.dtype))
    return torch.matmul(attn_weights, v)


@scaled_dot_product_attention.override(
    AnyTensor, AnyTensor, AnyTensor, AnyType, impl_name="torch"
)
def scaled_dot_product_attention_torch(q, k, v, a, is_causal, scale, softcap, impl):
    if softcap is not None:
        return NotImplemented
    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    if a is not None:
        a = unbox_tensor(a)

    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=a, dropout_p=0.0, is_causal=is_causal, scale=scale
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
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyType,
    impl_name="sharktank",
)
def scaled_dot_product_flash_attention_sharktank(
    q, k, v, a, is_causal, scale, softcap, impl
):
    if softcap:
        return NotImplemented

    if is_causal and a is None:
        seq_len = q.shape[-2]
        a = (
            torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    if scale is None:
        scale = torch.scalar_tensor(1.0 / math.sqrt(q.shape[-1]), dtype=torch.float32)
    else:
        scale = torch.scalar_tensor(scale, dtype=torch.float32)

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

    if a is not None:
        if a.dim() == 4:
            # TODO: Multiple tests are relying on inconsistent behavior of the attention mask.
            # Attention mask ranks should be consistent.
            # assert a.shape[0] == 1 and a.shape[1] == 1
            a = a[0, 0, :, :]
        atten = kernels.masked_flash_attention(q, k, v, a, scale)
    else:
        atten = kernels.flash_attention(q, k, v, scale)

    atten = atten * vscale if vscale is not None else atten
    return atten
