# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from amdsharktank.layers.configs import ModelConfig, ExportFunctionConfig
from amdsharktank.layers import BaseLayer, create_model
from amdsharktank.types import Dataset, Theta


class Model(BaseLayer):
    def __init__(self, config: ModelConfig):
        pass

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return ModelConfig


def test_create_model_from_config(tmp_path: Path):
    config = ModelConfig(
        model_type=Model,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)
    model = create_model(config_path)
    assert isinstance(model, Model)


def test_save_load_model_config(tmp_path: Path):
    config = ModelConfig(
        model_type=Model,
        config_path=f"{tmp_path / 'config.json'}",
        mlir_path="model.mlir",
        parameters_path="model.irpa",
        iree_module_path="model.vmfb",
        export_functions=[ExportFunctionConfig(function="forward", batch_sizes=[1])],
    )
    config.save()
    Dataset(properties={}, root_theta=Theta(tensors={})).save(config.parameters_path)
    config2 = ModelConfig.load(config.config_path)
    assert config == config2
