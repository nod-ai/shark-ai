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
from sharktank.layers.testing import make_random_moe_block_theta
from sharktank.utils.random import make_rand_torch
from sharktank.utils.testing import assert_tensor_close
from sharktank.layers.mixture_of_experts_block import MoeBlock
from sharktank.types.sharding import MoeBlockSharding
from sharktank.ops import reshard, reshard_like, replicate
from sharktank.types import unbox_tensor


class MoeBlockTest(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123)

    def testExport(self):
        dtype = torch.float32
        batch_size = 3
        seq_len = 5
        in_dim = 7

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=in_dim,
            expert_hidden_dim=13,
            num_experts=17,
            with_ffn_norm=True,
            num_shared_experts=19,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )
        theta.rename_tensors_to_paths()
        model = MoeBlock(
            theta=theta,
            expert_count=17,
            expert_used_count=2,
            rms_epsilon=1e-5,
        )
        fxb = FxProgramsBuilder(model)
        input = make_rand_torch((batch_size, seq_len, in_dim))

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
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
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
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
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
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
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
                batch_size=2,
                sequence_length=3,
                rms_epsilon=0.03,
                moe_activation_fn=torch.nn.functional.gelu,
                score_experts_fn=torch.nn.functional.softmax,
                normalize_experts=True,
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
                batch_size=17,
                sequence_length=19,
                rms_epsilon=0.01,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=False,
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
        batch_size: int,
        sequence_length: int,
        rms_epsilon: float,
        moe_activation_fn: Callable[[torch.Tensor], torch.Tensor],
        score_experts_fn: Callable[[torch.Tensor], torch.Tensor],
        normalize_experts: bool,
        route_scale: float,
    ):
        from sharktank.layers.testing import make_random_moe_block_theta
        from sharktank.layers import MoeBlock

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
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
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        res_pre_gather = moe_with_pre_gather_ffn(input)
        res_dense = moe_with_dense_ffn(input)
        assert_tensor_close(res_pre_gather, res_dense)

    @parameterized.expand(
        [
            param(
                dtype=torch.float32,
                feature_dim=7,
                expert_hidden_dim=3,
                num_experts=12,
                n_expert_groups=4,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=5,
                batch_size=8,
                sequence_length=9,
                rms_epsilon=0.01,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=None,
                tensor_parallelism_size=2,
            ),
            param(
                dtype=torch.bfloat16,
                feature_dim=2,
                expert_hidden_dim=10,
                num_experts=9,
                n_expert_groups=3,
                n_limited_groups=3,
                expert_used_count=7,
                num_shared_experts=8,
                batch_size=2,
                sequence_length=3,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.gelu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=1.1,
                tensor_parallelism_size=3,
            ),
        ]
    )
    def testTensorParallel(
        self,
        dtype: torch.dtype,
        feature_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        n_expert_groups: int | None,
        n_limited_groups: int | None,
        expert_used_count: int,
        num_shared_experts: int,
        batch_size: int,
        sequence_length: int,
        rms_epsilon: float,
        moe_activation_fn: Callable[[torch.Tensor], torch.Tensor],
        score_experts_fn: Callable[[torch.Tensor], torch.Tensor],
        normalize_experts: bool,
        route_scale: float,
        tensor_parallelism_size: int,
    ):
        from sharktank.layers.testing import make_random_moe_block_theta
        from sharktank.layers import MoeBlock

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=False,
            num_shared_experts=num_shared_experts,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )
        model_arch = "grok"
        if num_shared_experts > 0:
            model_arch = "deepseek2"
        theta_sharding_spec = MoeBlockSharding(
            shard_count=tensor_parallelism_size, model_arch=model_arch
        )
        sharded_theta = reshard(theta, spec=theta_sharding_spec)

        block = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )
        sharded_block = MoeBlock(
            theta=sharded_theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        sharded_input = replicate(input, count=tensor_parallelism_size)
        expected = block(input)
        actual = sharded_block(sharded_input)
        assert_tensor_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
