# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from sharktank import ops

import warnings

__all__ = [
    'attention',
]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
    attn_mask = None

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    '''
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
    dropout_p is for training
    '''
    out = ops.scaled_dot_product_attention(
        q, k, v, a=attn_mask, is_causal=causal)

    out = out.transpose(1, 2).contiguous()
    return out