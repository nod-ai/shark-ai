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


def _prepare_sink_tensor(
    sink: torch.Tensor, bs: int, n_heads: int, n_tokens: int
) -> torch.Tensor:
    """Prepare sink tensor for attention: [sink_size, n_heads] -> [bs, n_heads, n_tokens, sink_size]"""
    if sink.dim() == 1:
        # (n_heads) -> (bs, n_heads, n_tokens, 1)
        assert (
            sink.shape[0] == n_heads
        ), f"1D sink must have {n_heads} elements, got {sink.shape[0]}"
        return sink.reshape(1, n_heads, 1, 1).expand(bs, n_heads, n_tokens, 1)

    elif sink.dim() == 2:
        # Ensure shape is [n_heads, sink_size]
        if sink.shape[1] == n_heads and sink.shape[0] != n_heads:
            sink = sink.T  # transpose if needed
        assert (
            sink.shape[0] == n_heads
        ), f"Sink must have {n_heads} heads, got shape {sink.shape}"

        sink_size = sink.shape[1]
        return sink.reshape(1, n_heads, 1, sink_size).expand(
            bs, n_heads, n_tokens, sink_size
        )

    else:
        raise ValueError(f"Sink must be 1D or 2D, got {sink.dim()}D tensor")


# These two versions should be preserved in this order
@scaled_dot_product_attention.override(
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyType,
    impl_name="decomposed",
)
def scaled_dot_product_attention_decomposed(
    q, k, v, a, sink, is_causal, scale, softcap, impl
):

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    bs, n_heads, n_tokens, head_dim = q.shape

    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    attn_weights = attn_weights * scale
    if softcap is not None:
        attn_weights = softcap * torch.tanh(attn_weights / softcap)

    if a is not None:
        attn_weights = attn_weights + a

    if sink is not None:
        sink = _prepare_sink_tensor(sink, bs, n_heads, n_tokens)

        attn_weights = ops.cat([attn_weights, sink], dim=-1)
        attn_weights = ops.softmax(attn_weights, dim=-1)[..., : -sink.shape[-1]]
    else:
        attn_weights = ops.softmax(attn_weights, dim=-1)

    attn_weights = unbox_tensor(attn_weights)
    out = torch.matmul(attn_weights, v)

    return out.to(q.dtype)


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
    q, k, v, a, sink, is_causal, scale, softcap, impl
):
    if sink is not None:
        return NotImplemented
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


@scaled_dot_product_attention.override(
    AnyTensor, AnyTensor, AnyTensor, AnyType, impl_name="torch"
)
def scaled_dot_product_attention_torch(
    q, k, v, a, sink, is_causal, scale, softcap, impl
):
    if sink is not None:
        return NotImplemented
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
