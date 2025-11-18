# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

import math
import torch

from amdsharktank import kernels, ops
from amdsharktank.kernels.wave.extend_attention import wave_extend_attention
from amdsharktank.types import (
    AnyTensor,
    PlanarQuantizedTensor,
)

from amdsharktank.types.layouts import TensorScaledLayout

from amdsharktank.types.tensors import unbox_tensor
from .signatures import (
    scaled_dot_product_attention,
    extend_attention,
)
from ._registry import AnyType


def build_causal_and_sw_prefill(mask_prefill, n_tokens, sliding_window, dtype, device):
    if mask_prefill is None:
        mask_prefill = torch.triu(
            ops.full((n_tokens, n_tokens), -float("inf"), dtype=dtype, device=device),
            diagonal=1,
        )

    if sliding_window > 0:
        mask_prefill = mask_prefill + torch.tril(
            ops.full((n_tokens, n_tokens), -float("inf"), dtype=dtype, device=device),
            diagonal=-sliding_window,
        )
    return mask_prefill


def create_mask_sliding_window(
    a, attn_weights, sliding_window, n_tokens, kv_size, dtype, device
):
    if sliding_window is None or sliding_window <= 0:
        if a is not None:
            attn_weights = attn_weights + a
        return attn_weights

    is_prefill = kv_size == n_tokens
    if is_prefill:
        a = build_causal_and_sw_prefill(
            mask_prefill=a,
            n_tokens=n_tokens,
            sliding_window=(sliding_window or 0),
            device=device,
            dtype=dtype,
        )

    else:

        if sliding_window > 0 and kv_size > sliding_window:
            start_idx = kv_size - sliding_window
            neg_inf = float("-inf")

            # Apply sliding window: mask out tokens before start_idx
            a[..., :start_idx] = neg_inf

    if a is not None:
        attn_weights = attn_weights + a
    return attn_weights


def create_mask(a, attn_weights, is_causal):
    if a is not None:
        # Ensure mask has same dtype as attn_weights to avoid dtype promotion
        if a.dtype != attn_weights.dtype:
            raise ValueError("Incompatible tensor dtypes")
        attn_weights = attn_weights + a
    elif is_causal:
        mask = ops.full(
            (attn_weights.shape[2], attn_weights.shape[3]),
            float("-inf"),
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        attn_weights = attn_weights + mask
    return attn_weights


# These two versions should be preserved in this order
@scaled_dot_product_attention.override(
    AnyTensor,
    AnyTensor,
    AnyTensor,
    AnyType,
    impl_name="decomposed",
)
def scaled_dot_product_attention_decomposed(
    q, k, v, a, sink, sliding_window, is_causal, scale, softcap, impl
):

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)

    bs, n_heads, n_tokens, head_dim = q.shape
    kv_size = k.shape[-2]

    attn_weights = ops.matmul(q, k.transpose(-2, -1))
    attn_weights = attn_weights * scale
    if softcap is not None:
        attn_weights = softcap * torch.tanh(attn_weights / softcap)

    use_sink_path = (sink is not None) or (sliding_window is not None)
    if not use_sink_path:
        # standard causal/masked attention
        attn_weights = create_mask(a, attn_weights, is_causal)
        attn_weights = ops.softmax(attn_weights, dim=-1)
        out = ops.matmul(attn_weights, v)
        return out

    attn_weights = create_mask_sliding_window(
        a,
        attn_weights=attn_weights,
        n_tokens=n_tokens,
        kv_size=kv_size,
        sliding_window=sliding_window,
        dtype=q.dtype,
        device=q.device,
    )

    if sink is not None:
        max_attn_weights = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        lse = max_attn_weights + ops.log(
            torch.sum(torch.exp(attn_weights - max_attn_weights), dim=-1, keepdim=True)
        )
        lse = lse.squeeze(-1)

        attn_weights = ops.softmax(attn_weights, dim=-1)

        sink_expanded = sink.view(1, -1, 1)
        alpha = ops.sigmoid(lse - sink_expanded)
        result = ops.matmul(attn_weights, v) * alpha.unsqueeze(-1)
        return result

    attn_weights = ops.softmax(attn_weights, dim=-1)
    out = ops.matmul(attn_weights, v)
    return out


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
    impl_name="amdsharktank",
)
def scaled_dot_product_flash_attention_amdsharktank(
    q, k, v, a, sink, sliding_window, is_causal, scale, softcap, impl
):
    if sliding_window is not None and sliding_window > 0:
        return NotImplemented
    if sink is not None:
        return NotImplemented
    if softcap:
        return NotImplemented

    if is_causal and a is None:
        seq_len = q.shape[-2]
        a = (
            torch.triu(ops.full((seq_len, seq_len), float("-inf")), diagonal=1)
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
        a = unbox_tensor(a)
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
    q, k, v, a, sink, sliding_window, is_causal, scale, softcap, impl
):
    if sliding_window is not None and sliding_window > 0:
        return NotImplemented
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


@extend_attention.override(AnyTensor, AnyTensor, AnyTensor, impl_name="wave")
def extend_attention_wave(q, k, v, kv_cache, page_ids, start_positions, seq_lens, impl):
    if kv_cache is not None:
        return NotImplemented
    if page_ids is not None:
        return NotImplemented
    q = unbox_tensor(q)
    k = unbox_tensor(k)
    v = unbox_tensor(v)
    device = q.device
    B, L, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    _, _, _, D_kv = v.shape

    q_flat = q.flatten(0, 1).to(torch.float16).to(device)  # [B=1*extend_len, H_q, D]
    k_flat = k.flatten(0, 1).to(torch.float16).to(device)  # [B=1*extend_len, H_kv, D]
    v_flat = v.flatten(0, 1).to(torch.float16).to(device)
    k_cache = torch.zeros_like(k)
    v_cache = torch.zeros_like(v)
    k_cache_flat = (
        k_cache.flatten(0, 1).to(torch.float16).to(device)
    )  # [B*prefix_len, H_kv, D]
    v_cache_flat = v_cache.flatten(0, 1).to(torch.float16).to(device)
    extend_len = seq_lens - start_positions
    extend_len = extend_len.squeeze().to(dtype=torch.int32)
    b_seq_len_extend = ops.full(
        (B,), extend_len.item(), dtype=torch.int32, device=device
    )
    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, dim=0)
    kv_indptr = torch.zeros(q.shape[0] + 1, dtype=torch.int32)
    kv_indices = torch.zeros(q.shape[0], dtype=torch.int32)
    N_q = q_flat.shape[0]
    output_buffer = torch.zeros((N_q, H_q, D_kv), dtype=torch.float16, device=device)

    extend_attention = wave_extend_attention(
        q_flat,
        k_flat,
        v_flat,
        k_cache_flat,
        v_cache_flat,
        qo_indptr,
        kv_indptr,
        kv_indices,
        output_buffer,
        extend_len,
    )
    extend_attention = extend_attention.view(B, L, H_q, D)
    extend_attention = extend_attention.permute(0, 2, 1, 3)
    return extend_attention
