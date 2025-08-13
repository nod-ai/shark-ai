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
import math

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
    def __init__(
        self,
        head_dim,
        rope_theta,
        rope_openweight: bool = False,
        interleaved: bool = True,
        yarn_beta_slow: float | None = None,
        yarn_beta_fast: float | None = None,
        yarn_factor: float | None = None,
        yarn_original_context_len: int | None = None,
    ):
        super().__init__()
        self._rotary = RotaryEmbeddingLayer(
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_openweight=rope_openweight,
            interleaved=interleaved,
            yarn_beta_slow=yarn_beta_slow,
            yarn_beta_fast=yarn_beta_fast,
            yarn_factor=yarn_factor,
            yarn_original_context_len=yarn_original_context_len,
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


# ---------------------------
# OpenWeight reference and tests
# ---------------------------
class ReferenceOpenWeightRotary(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(2).to(x.dtype)
        sin = sin.unsqueeze(2).to(x.dtype)
        x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = torch.cat((o1, o2), dim=-1)
        return out

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin_from_position(self, position_ids: torch.Tensor):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        angles = position_ids[:, :, None].to(torch.float32) * inv_freq[
            None, None, :
        ].to(torch.float32)
        cos = angles.cos() * concentration
        sin = angles.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._compute_cos_sin_from_position(position_ids)

        query = self._apply_rotary_emb(query, cos, sin)

        key = self._apply_rotary_emb(key, cos, sin)

        return query, key


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 3e-5, 1e-5),
        (torch.float16, 2e-3, 1e-3),
        (torch.bfloat16, 2e-3, 1e-3),
    ],
)
def test_rotary_openweight_interweaved(dtype: torch.dtype, atol: float, rtol: float):

    bs = 1
    length = 128
    heads = 8
    dims = 64

    rope_theta = 150000.0
    yarn_factor = 32.0
    yarn_beta_slow = 1.0
    yarn_beta_fast = 32.0
    yarn_original_context_len = 4096

    st_rotary = STRotaryEmbedding(
        head_dim=dims,
        rope_theta=rope_theta,
        interleaved=False,  # HF interweaved pairing
        rope_openweight=True,
        yarn_factor=yarn_factor,
        yarn_beta_slow=yarn_beta_slow,
        yarn_beta_fast=yarn_beta_fast,
        yarn_original_context_len=yarn_original_context_len,
    )

    ref_rotary = ReferenceOpenWeightRotary(
        head_dim=dims,
        base=rope_theta,
        scaling_factor=yarn_factor,
        ntk_alpha=yarn_beta_slow,
        ntk_beta=yarn_beta_fast,
        initial_context_length=yarn_original_context_len,
        dtype=dtype,
    )

    def test_prefill():
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)
        position_ids = torch.arange(0, length)[None, :].repeat(bs, 1)

        st_q, st_k = st_rotary(q, k, position_ids)
        ref_q, ref_k = ref_rotary(q, k, position_ids)

        torch.testing.assert_close(st_q, ref_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(st_k, ref_k, atol=atol, rtol=rtol)

    def test_decode():
        q = torch.randn(bs, 1, heads, dims, dtype=dtype)
        k = torch.randn(bs, 1, heads, dims, dtype=dtype)
        position_ids = torch.randint(0, length, (bs, 1))

        st_q, st_k = st_rotary(q, k, position_ids)
        ref_q, ref_k = ref_rotary(q, k, position_ids)

        torch.testing.assert_close(st_q, ref_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(st_k, ref_k, atol=atol, rtol=rtol)

    test_prefill()
    test_decode()
