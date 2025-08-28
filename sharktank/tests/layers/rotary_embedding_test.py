# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import math
import unittest
import torch

from parameterized import parameterized_class

from sharktank import ops
from sharktank.layers import CachedRotaryLayer, build_rotary_layer
from sharktank.types import AnyTensor, ReplicatedTensor


def validate(
    xq: AnyTensor,
    em: AnyTensor,
    rope_dims: float,
    rope_freq_base: float,
    interleaved: bool,
) -> None:
    xq = ops.unshard(xq)
    em = ops.unshard(em)

    # Initially we want to compute the lengths of each vector
    if interleaved:
        xq_01 = xq.unflatten(-1, (rope_dims // 2, 2))
        em_01 = em.unflatten(-1, (2, rope_dims // 2))
        em_01 = torch.transpose(em_01, -2, -1)
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


@parameterized_class(
    ("use_hf",),
    [(True,), (False,)],
)
class TestRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bs = 1
        self.rope_dims = 8
        self.heads = 1
        self.max_seqlen = 16
        self.rope_freq_base = 10000.0
        self.pipeline_stage_to_device_map: list[list[int]] | None = None

        self.xq = torch.rand(
            (self.bs, self.max_seqlen, self.heads, self.rope_dims),
            dtype=torch.float,
        )

    def run_and_test_layer(
        self, xq: AnyTensor, rotary_layer: CachedRotaryLayer
    ) -> None:
        em = rotary_layer(xt=xq)
        validate(
            xq=self.xq,
            em=em,
            rope_dims=self.rope_dims,
            rope_freq_base=self.rope_freq_base,
            interleaved=(not self.use_hf),
        )

    def create_rotary_layer(self) -> CachedRotaryLayer:
        return build_rotary_layer(
            rope_dimension_count=self.rope_dims,
            rope_freq_base=self.rope_freq_base,
            use_hf=self.use_hf,
            pipeline_stage_to_device_map=self.pipeline_stage_to_device_map,
        )

    def test_rotary_table_unsharded(self):
        default_layer = self.create_rotary_layer()
        self.run_and_test_layer(self.xq, default_layer)

    def test_rotary_table_replicated(self):
        self.pipeline_stage_to_device_map = [[0], [0], [1], [1]]

        default_layer = self.create_rotary_layer()
        for devices in self.pipeline_stage_to_device_map:
            xq = ReplicatedTensor(ts=[self.xq], devices=devices)
            self.run_and_test_layer(xq, default_layer)
