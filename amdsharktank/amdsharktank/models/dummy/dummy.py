# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import torch
from collections import OrderedDict

from ...layers import (
    ThetaLayer,
    ModelConfig,
    LinearLayer,
    register_model_config_preset,
    configure_default_export_compile,
)
from ...types import Theta, AnyTensor, DefaultPrimitiveTensor
from amdsharktank.utils.random import make_rand_torch

__all__ = [
    "DummyModel",
    "DummyModelConfig",
]


@dataclass(kw_only=True)
class DummyModelConfig(ModelConfig):
    def __post_init__(self):
        self.model_type = DummyModel
        super().__post_init__()

    linear_input_size: int = 2
    linear_output_size: int = 3


class DummyModel(ThetaLayer):
    """A simple model for testing."""

    def __init__(self, *, config: DummyModelConfig, theta: Theta | None = None):
        super().__init__(theta, config=config)
        self.linear = LinearLayer(theta=self.theta("linear"))

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        """Return the type of the config for this model."""
        return DummyModelConfig

    def generate_random_theta(self) -> Theta:
        rng_seed = self.config.rng_seed
        if rng_seed is None:
            rng_seed = 12345
        with torch.random.fork_rng():
            torch.random.manual_seed(rng_seed)
            return Theta(
                {
                    "linear.weight": DefaultPrimitiveTensor(
                        data=make_rand_torch(
                            shape=[
                                self.config.linear_output_size,
                                self.config.linear_input_size,
                            ]
                        )
                    )
                }
            )

    def forward(self, x: AnyTensor) -> AnyTensor:
        return self.linear(x)

    def sample_inputs(
        self, batch_size: int | None = 1, function: str | None = None
    ) -> tuple[tuple[AnyTensor, ...], OrderedDict[str, AnyTensor]]:
        assert batch_size is not None
        assert function == "forward" or function is None

        return tuple(), OrderedDict(
            [
                (
                    "x",
                    make_rand_torch(shape=[batch_size, self.config.linear_input_size]),
                )
            ]
        )


def _register_dummy_model_config_presets():
    config = DummyModelConfig()
    configure_default_export_compile(config)
    register_model_config_preset(
        name="dummy-model-local-llvm-cpu",
        config=config.asdict_for_saving(),
    )


_register_dummy_model_config_presets()
