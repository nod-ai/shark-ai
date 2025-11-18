# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import fire
import json
import sys

from ..layers import (
    model_config_presets,
    ModelConfig,
    create_model,
    BaseLayer,
    register_all_models,
)


class Model:
    """Command line tool for performing operations on amdsharktank models.

    Common parameters:

    config
      Path to a config file or name of a config preset. The file takes precedence if it
      exists.

    out_dir
      Path to the output directory where generated artifacts are placed.
    """

    def export(self, config: str, /):
        """Export a model. This includes MLIR and optionally sample input arguments."""
        model = self._resolve_model(config)
        model.export()

    def compile(self, config: str, /):
        """Compile a model.
        The model needs to be exported before compiling."""
        model = self._resolve_model(config)
        model.compile()

    def list(self):
        """List all model config preset names."""
        register_all_models()
        for name in model_config_presets.keys():
            print(name)

    def show(self, config: str):
        """Show model config."""
        model_config = self._resolve_config(config)
        json.dump(model_config, fp=sys.stdout, indent=2, sort_keys=True)
        print()

    def tracy_trace(self, config: str, /, function: str | None = None):
        """Trace model with tracy.
        If function is not given all model functions are traced."""
        raise NotImplementedError("TODO")

    def _resolve_config(self, config: str) -> ModelConfig:
        register_all_models()
        if Path(config).exists():
            return ModelConfig.load(config)
        return model_config_presets[config]

    def _resolve_model(self, config: str) -> BaseLayer:
        config = self._resolve_config(config)
        return create_model(config)


class Cli:
    def __init__(self):
        self.model = Model()


def main():
    fire.Fire(Cli())


if __name__ == "__main__":
    main()
