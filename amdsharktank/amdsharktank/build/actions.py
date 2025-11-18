# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable
from iree.build.executor import (
    ActionConcurrency,
    BuildAction,
    BuildContext,
    BuildFile,
    Executor,
)
from os import PathLike
import importlib

from ..layers import ModelConfig, get_model_type_id

__all__ = ["export_model"]

PickleableType = Any


def export_model(
    config: dict[str, PickleableType] | PathLike,
    *,
    concurrency: ActionConcurrency = ActionConcurrency.PROCESS,
) -> list[BuildFile]:
    """Export a model from config.
    This function is meant to be used in an IREE build pipeline."""
    if isinstance(config, dict):
        model_config = ModelConfig.create(**config)
    else:
        model_config = ModelConfig.load(config)

    desc = f"Export amdsharktank model {get_model_type_id(model_config.model_type)}"
    if model_config.config_path is not None:
        desc += f" with config {model_config.config_path}"
    elif model_config.mlir_path is not None:
        desc += f" into {model_config.mlir_path}"

    context = BuildContext.current()
    action = Action(
        desc=desc,
        concurrency=concurrency,
        executor=BuildContext.current().executor,
        thunk=Thunk(
            export_model_thunk,
            args=tuple(),
            kwargs={
                "config": config,
                "model_python_module": str(model_config.model_type.__module__),
            },
        ),
    )
    output_file_paths = [model_config.mlir_path]
    if model_config.parameters_path is not None:
        output_file_paths.append(model_config.parameters_path)
    output_files = []
    for p in output_file_paths:
        output_file = context.allocate_file(str(p.absolute()))
        output_file.deps.add(action)
        output_files.append(output_file)
    return output_files


def export_model_thunk(
    config: dict[str, PickleableType] | PathLike, model_python_module: str
):
    from ..layers import create_model

    # This is required in order for the model to get auto-registered when scheduling
    # execution in a subprocess. Without it model creation would fail.
    importlib.import_module(model_python_module)
    model = create_model(config)
    model.export()


class Thunk:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[PickleableType, ...],
        kwargs: dict[str, PickleableType],
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.fn(*self.args, **self.kwargs)


class Action(BuildAction):
    def __init__(
        self,
        desc: str,
        *,
        concurrency: ActionConcurrency,
        executor: Executor | None = None,
        thunk: Thunk,
    ):
        super().__init__(desc=desc, executor=executor, concurrency=concurrency)
        self.thunk = thunk

    def _remotable_thunk(self):
        return self.thunk
