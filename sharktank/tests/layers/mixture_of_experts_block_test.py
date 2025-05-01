# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from parameterized import parameterized, param
from typing import Callable

import torch
from iree.turbine.aot import *
from sharktank.models.llama.testing import make_moe_block_theta, make_rand_torch
from sharktank.layers.mixture_of_experts_block import MoeBlock


class MoeBlockTest(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123)

    def testExport(self):
        model = MoeBlock(
            theta=make_moe_block_theta()("blk.0"),
            expert_used_count=2,
            rms_epsilon=1e-5,
        )
        fxb = FxProgramsBuilder(model)
        input = make_rand_torch((2, 32, 6144))

        @fxb.export_program(name="moe_block", args=(input,), strict=False)
        def _(model, input: torch.Tensor) -> torch.Tensor:
            return model(input)

    @parameterized.expand(
        [
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=1,
                expert_used_count=1,
                n_expert_groups=None,
                n_limited_groups=None,
                num_shared_experts=1,
                shared_expert_hidden_dim=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                add_residual=False,
                route_scale=1.234,
            ),
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=2,
                n_expert_groups=None,
                n_limited_groups=None,
                expert_used_count=1,
                num_shared_experts=1,
                shared_expert_hidden_dim=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                add_residual=False,
                route_scale=1.234,
            ),
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=3,
                n_expert_groups=None,
                n_limited_groups=None,
                expert_used_count=2,
                num_shared_experts=1,
                shared_expert_hidden_dim=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                add_residual=False,
                route_scale=1.234,
            ),
            param(
                dtype=torch.float32,
                feature_dim=2,
                expert_hidden_dim=3,
                num_experts=4,
                n_expert_groups=2,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=2,
                shared_expert_hidden_dim=3,
                batch_size=2,
                sequence_length=3,
                rms_epsilon=0.03,
                moe_activation_fn=torch.nn.functional.gelu,
                score_experts_fn=torch.nn.functional.softmax,
                normalize_experts=True,
                add_residual=True,
                route_scale=3.21,
            ),
            param(
                dtype=torch.bfloat16,
                feature_dim=7,
                expert_hidden_dim=3,
                num_experts=12,
                n_expert_groups=3,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=11,
                shared_expert_hidden_dim=13,
                batch_size=17,
                sequence_length=19,
                rms_epsilon=0.01,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=False,
                add_residual=False,
                route_scale=None,
            ),
        ]
    )
    def testParityOfExpertPreGatherFfnAndDenseFfn(
        self,
        dtype: torch.dtype,
        feature_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        n_expert_groups: int | None,
        n_limited_groups: int | None,
        expert_used_count: int,
        num_shared_experts: int,
        shared_expert_hidden_dim: int,
        batch_size: int,
        sequence_length: int,
        rms_epsilon: float,
        moe_activation_fn: Callable[[torch.Tensor], torch.Tensor],
        score_experts_fn: Callable[[torch.Tensor], torch.Tensor],
        normalize_experts: bool,
        add_residual: bool,
        route_scale: float,
    ):
        from sharktank.layers.testing import make_random_moe_block_theta
        from sharktank.layers import MoeBlock

        theta = make_random_moe_block_theta(
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            shared_expert_hidden_dim=shared_expert_hidden_dim,
            with_layer_output_norm=True,
            dtype=dtype,
        )

        moe_with_pre_gather_ffn = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            experts_ffn_moe_block="PreGatherFFNMOE",
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            add_residual=add_residual,
            route_scale=route_scale,
        )
        moe_with_dense_ffn = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            experts_ffn_moe_block="DenseFFNMOE",
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            add_residual=add_residual,
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        res_pre_gather = moe_with_pre_gather_ffn(input)
        res_dense = moe_with_dense_ffn(input)
        torch.testing.assert_close(res_pre_gather, res_dense)


if __name__ == "__main__":
    unittest.main()
