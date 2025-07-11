# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
import unittest

from parameterized import parameterized_class
from sharktank.layers.rotary_embedding import build_rotary_layer
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig


class ValidateRotaryEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123456)

    def validate(self, xq, em, rope_dims, rope_freq_base, interleaved):
        # Initially we want to compute the lengths of each vector
        if interleaved:
            xq_01 = xq.unflatten(-1, (rope_dims // 2, 2))
            em_01 = em.unflatten(-1, (rope_dims // 2, 2))
        else:
            xq_01 = xq.unflatten(-1, (2, rope_dims // 2))
            em_01 = em.unflatten(-1, (2, rope_dims // 2))
            xq_01 = torch.transpose(xq_01, -2, -1)
            em_01 = torch.transpose(em_01, -2, -1)

        xq_0 = xq_01[:, :, :, :, 0]
        xq_1 = xq_01[:, :, :, :, 1]

        em_0 = em_01[:, :, :, :, 0]
        em_1 = em_01[:, :, :, :, 1]

        xq_l = torch.sqrt(xq_0 * xq_0 + xq_1 * xq_1)
        em_l = torch.sqrt(em_0 * em_0 + em_1 * em_1)
        torch.testing.assert_close(xq_l, em_l)

        # Normalize
        xq_0 = xq_0 / xq_l
        xq_1 = xq_1 / xq_l
        em_0 = em_0 / em_l
        em_1 = em_1 / em_l

        # Compute the angle step per value
        xq_a = torch.atan2(xq_1, xq_0)
        em_a = torch.atan2(em_1, em_0)

        # Compute the step size for the rotation
        angle = em_a - xq_a
        angle = angle[:, 1:, :, :] - angle[:, :-1, :, :]
        step = angle[0, 1, 0, :][None, None, None, :]
        step = torch.where(step > math.pi * 2.0, step - math.pi * 2.0, step)
        step = torch.where(step < 0.0, step + math.pi * 2.0, step)

        # Check that the step size is approximately correct
        expected_step = torch.log(torch.asarray(rope_freq_base)) * (
            -(torch.arange(rope_dims // 2)) / (rope_dims // 2)
        )
        expected_step = torch.exp(expected_step)
        torch.testing.assert_close(step.flatten(), expected_step, atol=1e-2, rtol=1e-2)

        # Guarantee a progressive stepping for rotation:
        angle = angle / step
        angle = angle[:, 1:, ::]
        angle = torch.where(angle < 0, angle + math.pi * 2.0, angle)
        torch.testing.assert_close(
            angle, torch.full(angle.shape, 1.0), atol=1e-2, rtol=1e-2
        )

    def test_sharded_rotary_table_interleaved(self):
        bs = 1
        rope_dims = 8
        heads = 1
        max_seqlen = 16
        rope_freq_base = 10000.0

        # First we setup and get the default rotary embedding layer
        xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
        default_layer = build_rotary_layer(
            rope_dimension_count=rope_dims,
            max_seqlen=max_seqlen,
            rope_freq_base=rope_freq_base,
            use_hf=False,
        )
        em = default_layer(xt=xq, start_index=0)
        self.validate(
            xq=xq,
            em=em,
            rope_dims=rope_dims,
            rope_freq_base=rope_freq_base,
            interleaved=True,
        )

    def test_sharded_rotary_table_concatted(self):
        bs = 1
        rope_dims = 8
        heads = 1
        max_seqlen = 16
        rope_freq_base = 10000.0

        # First we setup and get the default rotary embedding layer
        xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
        default_layer = build_rotary_layer(
            rope_dimension_count=rope_dims,
            max_seqlen=max_seqlen,
            rope_freq_base=rope_freq_base,
            use_hf=True,
        )
        em = default_layer(xt=xq, start_index=0)
        self.validate(
            xq=xq,
            em=em,
            rope_dims=rope_dims,
            rope_freq_base=rope_freq_base,
            interleaved=False,
        )


class HFRotaryEmbedding(torch.nn.Module):
    def __init__(self, hf_config):
        super().__init__()
        self._rotary = LlamaRotaryEmbedding(config=hf_config)

    def forward(self, *, xt, positions):
        cos, sin = self._rotary(xt, positions)
        xt = xt.transpose(1, 2)
        return apply_rotary_pos_emb(xt, xt, cos, sin)[0].transpose(1, 2)


class HFRotaryComparisonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123456)

        self.rope_scaling = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        self.hf_config = LlamaConfig(
            rope_scaling=self.rope_scaling,
            max_position_embeddings=131072,
            rope_theta=500000,
        )

    def test_decode(self):
        test_dtype = torch.bfloat16
        bs = 2
        length = 5
        heads = 3
        dims = 128

        st_rotary = build_rotary_layer(
            rope_dimension_count=dims,
            max_seqlen=2048,
            rope_freq_base=500000,
            use_hf=True,
            dtype=test_dtype,
            yarn_beta_slow=1,
            yarn_beta_fast=4,
            yarn_factor=8,
            yarn_original_context_len=8192,
        )

        hf_rotary = HFRotaryEmbedding(self.hf_config)

        example = torch.rand(bs, length, heads, dims, dtype=test_dtype)
        positions = torch.arange(0, length)[None, :].repeat(bs, 1)

        decode_example = torch.rand(bs, 1, heads, dims, dtype=test_dtype)
        mask = st_rotary.compute_batch_mask(
            start_positions=torch.arange(0, bs), batch_seq_len=1
        )
        st_results = st_rotary.apply_batched_mask(xt=decode_example, mask=mask)
        hf_results = hf_rotary.forward(
            xt=decode_example, positions=torch.arange(0, bs).unsqueeze(1)
        )
        assert torch.all(torch.eq(st_results, hf_results))

        hf_results = hf_rotary(xt=example, positions=positions)
        st_results = st_rotary.forward(xt=example, start_index=0)
        assert torch.all(torch.eq(st_results, hf_results))

    def test_prefill(self):
        test_dtype = torch.float32
        bs = 2
        length = 10
        batch_seq_len = 12
        heads = 3
        dims = 128

        st_rotary = build_rotary_layer(
            rope_dimension_count=dims,
            max_seqlen=2048,
            rope_freq_base=500000,
            use_hf=True,
            dtype=test_dtype,
            yarn_beta_slow=1,
            yarn_beta_fast=4,
            yarn_factor=8,
            yarn_original_context_len=8192,
        )

        hf_rotary = HFRotaryEmbedding(self.hf_config)

        example = torch.rand(bs, length, heads, dims, dtype=test_dtype)
        prefill_example = torch.rand(bs, batch_seq_len, heads, dims, dtype=test_dtype)
        positions = torch.arange(0, length)[None, :].repeat(bs, 1)

        mask = st_rotary.compute_batch_mask(
            start_positions=torch.arange(0, bs), batch_seq_len=batch_seq_len
        )
        st_results = st_rotary.apply_batched_mask(xt=prefill_example, mask=mask)
        start_pos = torch.arange(0, bs)
        hf_positions = (
            torch.arange(0, batch_seq_len).unsqueeze(0).repeat(bs, 1)
            + start_pos[:, None]
        )
        hf_results = hf_rotary.forward(
            xt=prefill_example,
            positions=hf_positions,
        )
        assert torch.all(torch.eq(st_results, hf_results))

        hf_results = hf_rotary(xt=example, positions=positions)
        st_results = st_rotary.forward(xt=example, start_index=0)
        assert torch.all(torch.eq(st_results, hf_results))


@parameterized_class(
    ("use_table",),
    [
        {"use_table": True},
        {"use_table": False},
    ],
)
class SharktankRotaryMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123456)

        self.bs = 2
        self.heads = 1
        self.dims = 128
        self.max_seqlen = 16
        self.dtype = torch.float32

        self.rope = build_rotary_layer(
            rope_dimension_count=self.dims,
            max_seqlen=self.max_seqlen,
            rope_freq_base=10000,
            use_hf=False,
            use_table=self.use_table,
            dtype=self.dtype,
            yarn_beta_slow=1,
            yarn_beta_fast=1,
            yarn_factor=1,
            yarn_original_context_len=self.max_seqlen,
        )

        self.table = self.rope.rotary_embed_table

    def test_decode(self):
        batch_seq_len = 1

        xt = torch.rand(self.bs, batch_seq_len, self.heads, self.dims, dtype=self.dtype)
        start_positions = torch.tensor([0, 3], dtype=torch.int32)

        mask = self.rope.compute_batch_mask(
            start_positions=start_positions,
            batch_seq_len=batch_seq_len,
        )
        out_mask = self.rope.apply_batched_mask(xt=xt, mask=mask)

        inv_freq = torch.exp(
            -torch.arange(0, self.dims // 2, dtype=self.dtype)
            * math.log(self.rope._rotary_layer.rope_freq_base)
            / (self.dims // 2)
        )

        angles = start_positions[:, None].to(self.dtype) * inv_freq[None, :]

        cos = torch.cos(angles).view(self.bs, 1, 1, -1)
        sin = torch.sin(angles).view(self.bs, 1, 1, -1)

        x0 = xt[..., 0::2]
        x1 = xt[..., 1::2]

        out0 = x0 * cos - x1 * sin
        out1 = x1 * cos + x0 * sin

        out_expected = torch.stack([out0, out1], dim=-1).flatten(-2)

        torch.testing.assert_close(
            out_mask,
            out_expected,
            atol=1e-6,
            rtol=1e-5,
        )

    def test_prefill(self):
        batch_seq_len = 12

        xt = torch.rand(self.bs, batch_seq_len, self.heads, self.dims, dtype=self.dtype)
        start_positions = torch.tensor([0, 3], dtype=torch.int32)

        mask = self.rope.compute_batch_mask(
            start_positions=start_positions,
            batch_seq_len=batch_seq_len,
        )
        out_mask = self.rope.apply_batched_mask(xt=xt, mask=mask)

        inv_freq = torch.exp(
            -torch.arange(0, self.dims // 2, dtype=self.dtype)
            * math.log(self.rope._rotary_layer.rope_freq_base)
            / (self.dims // 2)
        )
        positions = (
            start_positions[:, None].to(self.dtype)
            + torch.arange(batch_seq_len, dtype=self.dtype)[None, :]
        )
        angles = positions[..., None] * inv_freq[None, None, :]

        cos = torch.cos(angles).view(self.bs, batch_seq_len, 1, -1)
        sin = torch.sin(angles).view(self.bs, batch_seq_len, 1, -1)

        x0 = xt[..., 0::2]
        x1 = xt[..., 1::2]

        out0 = x0 * cos - x1 * sin
        out1 = x1 * cos + x0 * sin

        out_expected = torch.stack([out0, out1], dim=-1).flatten(-2)

        torch.testing.assert_close(
            out_mask,
            out_expected,
            atol=1e-6,
            rtol=1e-5,
            msg="Sharktank prefill route (`batch_seq_len=4`) mismatch",
        )
