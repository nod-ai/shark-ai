# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from typing import Any
from sharktank import ops
from sharktank.layers import ThetaLayer, LinearLayer
from sharktank.transforms.pipeline_parallelism import split_module, SplitPoint
from sharktank.types import AnyTensor, DefaultPrimitiveTensor, Theta
from sharktank.types.sharding import (
    LinearSplitReductionDimSharding,
    LinearReplicatedWeightAndBiasSharding,
    ThetaSharding,
)
from sharktank.utils.testing import assert_tensor_close


class SampleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(3, 3) for _ in range(2))
        self.lm = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
            x = x + y
            y = x * x
        z = self.lm(x)
        return x, z

    def sample_args(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (torch.LongTensor([1, 2, 4, 5]),), {
            "y": torch.rand([4, 3], dtype=torch.float32)
        }


@pytest.fixture(scope="function")
def sample_module() -> SampleModule:
    return SampleModule()


class SampleSharktankModule(ThetaLayer):
    def __init__(self, theta: Theta, linear_count: int):
        super().__init__(theta)
        self.layers = torch.nn.Sequential(
            *[LinearLayer(theta(f"layers.{i}")) for i in range(linear_count)]
        )

    def forward(self, x: AnyTensor, y: AnyTensor) -> AnyTensor:
        self.layers(x)
        z = x + y
        return z

    def sample_args(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # shard_count = self.theta("layers.0.weight").shard_count
        x = torch.rand(
            [
                4,
                self.theta("layers.1.weight").shape[1],
                self.theta("layers.0.weight").shape[1],
            ],
            dtype=torch.float32,
        )
        # x = ops.reshard_split(x, dim=1, count=shard_count)
        y = torch.rand(
            [
                4,
                self.theta(f"layers.{len(self.layers) - 1}.weight").shape[1],
                self.theta(f"layers.{len(self.layers) - 1}.weight").shape[0],
            ],
            dtype=torch.float32,
        )
        # y = ops.replicate(y, count=shard_count)
        return (x,), {"y": y}


@pytest.fixture(scope="function")
def sample_sharktank_module() -> SampleSharktankModule:
    shard_count = 1
    in_features = shard_count * 3
    out_features = shard_count * 5

    theta = Theta(
        {
            "layers.0.weight": DefaultPrimitiveTensor(
                data=torch.rand(out_features, in_features, dtype=torch.float32)
            ),
            "layers.1.weight": DefaultPrimitiveTensor(
                data=torch.rand(in_features, out_features, dtype=torch.float32)
            ),
            "layers.1.bias": DefaultPrimitiveTensor(
                data=torch.rand(in_features, dtype=torch.float32)
            ),
        }
    )
    # theta_sharding = ThetaSharding(
    #         {
    #             "layer.0": LinearSplitReductionDimSharding(
    #                 shard_count=shard_count
    #             ).theta_sharding(),
    #             "layer.1": LinearReplicatedWeightAndBiasSharding(shard_count=0)
    #         }
    #     )
    # theta = ops.reshard(theta, theta_sharding)
    # theta = Theta({
    #     "layers.0.weight": ops.replicate(theta("layers.0.weight"), count=shard_count, devices=(0, 1)),
    #     "layers.1.weight": ops.reshard_split(theta("layers.1.weight"), dim=1, count=shard_count, devices=(2, 3)),
    #     "layers.1.bias": ops.replicate(theta("layers.1.bias"), count=shard_count, devices=(2, 3)),
    # })

    return SampleSharktankModule(theta, linear_count=2)


def sample_sharktank_module_pipeline_parallel_theta(theta: Theta):
    return Theta(
        {
            "layers.0.weight": ops.replicate(
                theta("layers.0.weight"), count=1, devices=(0,)
            ),
            "layers.1.weight": ops.replicate(
                theta("layers.1.weight"), count=1, devices=(1,)
            ),
            "layers.1.bias": ops.replicate(
                theta("layers.1.bias"), count=1, devices=(1,)
            ),
        }
    )


class TestPipelineParallelism:
    def test_split_into_stages(
        self, deterministic_random_seed, sample_module: SampleModule
    ):
        args, kwargs = sample_module.sample_args()
        expected_results = sample_module(*args, **kwargs)

        pipe = split_module(
            sample_module,
            split_spec={
                ("layers.1", "forward"): SplitPoint.BEGINNING,
            },
            args=args,
            kwargs=kwargs,
        )
        assert pipe.num_stages == 2
        actual_results = pipe.split_gm(*args, **kwargs)

        assert_tensor_close(actual_results, expected_results, rtol=0, atol=0)

    def test_split_into_stages_sharktank_module(
        self, deterministic_random_seed, sample_sharktank_module: SampleSharktankModule
    ):
        module = sample_sharktank_module
        args, kwargs = module.sample_args()
        # Sanity check that the module runs without errors.
        expected_results = module(*args, **kwargs)

        module_with_pp_theta = SampleSharktankModule(
            theta=sample_sharktank_module_pipeline_parallel_theta(module.theta),
            linear_count=len(module.layers),
        )
        pipe = split_module(
            module_with_pp_theta,
            split_spec={
                ("layers.1", "forward"): SplitPoint.BEGINNING,
            },
            args=args,
            kwargs=kwargs,
        )
        assert pipe.num_stages == 2
        actual_results = pipe.split_gm(*args, **kwargs)

        assert_tensor_close(actual_results, expected_results, rtol=0, atol=0)
