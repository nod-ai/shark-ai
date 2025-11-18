# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict
from iree.build import compile, entrypoint, iree_build_main
import os
from dataclasses import dataclass
import torch

from amdsharktank.layers import ThetaLayer, ModelConfig, LinearLayer
from amdsharktank.types import Theta, DefaultPrimitiveTensor
from amdsharktank.build import export_model
from amdsharktank.utils.random import make_rand_torch


@dataclass(kw_only=True)
class ExampleModelConfig(ModelConfig):
    in_size: int
    out_size: int


class ExampleModel(ThetaLayer):
    def __init__(self, theta: Theta | None = None, config: ModelConfig | None = None):
        super().__init__(config=config, theta=theta)
        self.linear = LinearLayer(theta=self.theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return ExampleModelConfig

    def generate_random_theta(self) -> Theta:
        return Theta(
            {
                "weight": DefaultPrimitiveTensor(
                    data=make_rand_torch(
                        shape=[self.config.out_size, self.config.in_size]
                    ),
                    name="weight",
                ),
                "bias": DefaultPrimitiveTensor(
                    data=make_rand_torch(shape=[1, self.config.out_size]), name="bias"
                ),
            }
        )

    def sample_inputs(self, batch_size: int = 1, function: str | None = None):
        return tuple(), OrderedDict(
            [("x", make_rand_torch(shape=[batch_size, self.config.in_size]))]
        )


@entrypoint(description="Pipeline to build an example model")
def pipe():
    export_model(
        ExampleModelConfig(
            model_type=ExampleModel,
            mlir_path="model.mlir",
            export_parameters_path="model.irpa",
            in_size=3,
            out_size=4,
        ).asdict_for_saving(),
    ),
    # TODO: add compile action that consumes a config.
    return compile(
        name=f"model",
        source=os.path.abspath("model.mlir"),
    )


if __name__ == "__main__":
    iree_build_main()
