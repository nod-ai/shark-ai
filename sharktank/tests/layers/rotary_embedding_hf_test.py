# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from sharktank.layers.rotary_embedding_hf import RotaryEmbeddingLayer
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig
import pytest

torch.manual_seed(123456)


class HFRotaryEmbedding(torch.nn.Module):
    def __init__(self, config, interleaved: bool = True):
        super().__init__()
        self._rotary = LlamaRotaryEmbedding(config=config)
        self.interleaved = interleaved

    def forward(self, q, k, positions):
        cos, sin = self._rotary(q, positions)
        dim = q.shape[-1]
        if self.interleaved:
            q = q.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
            k = k.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        if self.interleaved:
            q = q.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
            k = k.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
        return q, k


class STRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, rope_theta, interleaved: bool = True):
        super().__init__()
        self._rotary = RotaryEmbeddingLayer(
            head_dim=head_dim, rope_theta=rope_theta, interleaved=interleaved
        )

    def forward(self, q, k, positions):
        cossin_cache = self._rotary.compute_sincos_cache(positions, q.dtype)
        q = self._rotary(q, cossin_cache)
        k = self._rotary(k, cossin_cache)
        return (q, k)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_rotary_interweaved(dtype: torch.dtype):
    bs = 2
    length = 256
    heads = 16
    dims = 128

    hf_config = LlamaConfig(
        max_position_embeddings=131072,
        rope_theta=500000,
    )

    hf_rotary = HFRotaryEmbedding(hf_config, interleaved=False)

    st_rotary = STRotaryEmbedding(head_dim=dims, rope_theta=500000, interleaved=False)

    def test_prefill():
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)
        position_ids = torch.arange(0, length)[None, :].repeat(bs, 1)
        hf_results = hf_rotary(q, k, position_ids)
        st_results = st_rotary(q, k, position_ids)
        torch.testing.assert_close(hf_results, st_results)

    def test_decode():
        q = torch.randn(bs, 1, heads, dims)
        k = torch.randn(bs, 1, heads, dims)
        position_ids = torch.randint(0, length, (bs, 1))
        hf_results = hf_rotary(q, k, position_ids)
        st_results = st_rotary(q, k, position_ids)
        torch.testing.assert_close(hf_results, st_results)

    test_prefill()
    test_decode()


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2e-5, 1e-5),
        (torch.float16, None, None),
        (torch.bfloat16, None, None),
    ],
)
def test_rotary_interleaved(dtype: torch.dtype, atol: float, rtol: float):
    bs = 2
    length = 256
    heads = 16
    dims = 128

    hf_config = LlamaConfig(
        max_position_embeddings=131072,
        rope_theta=500000,
    )

    hf_rotary = HFRotaryEmbedding(hf_config, interleaved=True)

    st_rotary = STRotaryEmbedding(head_dim=dims, rope_theta=500000, interleaved=True)

    # Sharktank RoPE implementation does permutation along the reduction
    # dimension of Q @ K.T matmul, and is only correct post Q @ K.T matmul.
    # The HF implementation also relies on this, which is why you will notice
    # we do the unflatten + transpose + flatten post hf_rotary application.
    def rot_and_qk(rot, q, k, position_ids):
        q, k = rot(q, k, position_ids)
        q = q.transpose(1, 2).flatten(0, 1)
        k = k.transpose(1, 2).flatten(0, 1)
        out = q @ k.transpose(1, 2)
        return out

    def test_prefill():
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)
        position_ids = torch.arange(0, length)[None, :].repeat(bs, 1)
        leave = rot_and_qk(hf_rotary, q, k, position_ids)
        weave = rot_and_qk(st_rotary, q, k, position_ids)
        # Use a bigger atol because we are doing a matmul.
        torch.testing.assert_close(leave, weave, atol=atol, rtol=rtol)

    def test_decode():
        q = torch.randn(bs, 1, heads, dims, dtype=dtype)
        k = torch.randn(bs, 1, heads, dims, dtype=dtype)
        position_ids = torch.randint(0, length, (bs, 1))
        leave = rot_and_qk(hf_rotary, q, k, position_ids)
        weave = rot_and_qk(st_rotary, q, k, position_ids)
        # Use a bigger atol because we are doing a matmul.
        torch.testing.assert_close(leave, weave, atol=atol, rtol=rtol)

    test_prefill()
    test_decode()
