# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch

from sharktank.layers import RotaryEmbeddingLayer


def validate(xq, em, rope_dims, rope_freq_base, interleaved):
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


def test_sharded_rotary_table_interleaved():
    bs = 1
    rope_dims = 8
    heads = 1
    max_seqlen = 16
    rope_freq_base = 10000.0

    # First we setup and get the default rotary embedding layer
    xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    default_layer = RotaryEmbeddingLayer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
        use_hf=False,
    )
    em = default_layer(xt=xq, start_index=0)
    validate(
        xq=xq,
        em=em,
        rope_dims=rope_dims,
        rope_freq_base=rope_freq_base,
        interleaved=True,
    )


def test_sharded_rotary_table_concatted():
    bs = 1
    rope_dims = 8
    heads = 1
    max_seqlen = 16
    rope_freq_base = 10000.0

    # First we setup and get the default rotary embedding layer
    xq = torch.rand((bs, max_seqlen, heads, rope_dims), dtype=torch.float)
    default_layer = RotaryEmbeddingLayer(
        rope_dimension_count=rope_dims,
        max_seqlen=max_seqlen,
        rope_freq_base=rope_freq_base,
        use_hf=True,
    )
    em = default_layer(xt=xq, start_index=0)
    validate(
        xq=xq,
        em=em,
        rope_dims=rope_dims,
        rope_freq_base=rope_freq_base,
        interleaved=False,
    )
