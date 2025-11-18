# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import PathLike
from typing import Any, Dict, Optional
from collections import OrderedDict
from collections.abc import Mapping
from abc import ABCMeta
from pathlib import Path

import torch
import torch.nn as nn

from amdsharktank.types import InferenceTensor, Theta, AnyTensor, Dataset
from amdsharktank.utils import debugging, chdir
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.iree import flatten_for_iree_signature
from .configs import ModelConfig, ExportFunctionConfig, DynamicBatchSize

from iree.turbine.support.tools import iree_tool_prepare_input_args

__all__ = [
    "BaseLayer",
    "ThetaLayer",
    "create_model",
    "get_model_type_id",
    "model_registry",
    "register_all_models",
]

logger = get_logger(__name__)


def _set_recursively_submodules_default_trace_tensor_key_prefix(
    module: nn.Module, prefix: str = ""
):
    if isinstance(module, BaseLayer):
        module.trace_tensor_key_prefix = prefix

    for name, submodule in module.named_children():
        submodule_prefix = f"{prefix}{name}."
        _set_recursively_submodules_default_trace_tensor_key_prefix(
            submodule, submodule_prefix
        )


def get_model_type_id(model_type: type["BaseLayer"]) -> str:
    """Get a string representation of the model type."""
    return f"{model_type.__module__}.{model_type.__name__}"


def create_model(config: PathLike | ModelConfig | Mapping[str, Any], /) -> "BaseLayer":
    """
    Create model from a configuration.
    Example

    config.json:
    ```
    {
        "config_version": "0.1.0",
        "model_type": "MyModel",
        "mlir_path": "model.mlir",
        "export_parameters_path": "model.irpa",
        "iree_module_path": "model.vmfb",
        "compile_args": ["--iree-hal-target-device=local"],
        "export_functions": [
            {
                "function": "forward",
                "batch_sizes": [1, 2, 3]
            }
        ]
    ]
    }
    ```

    usage
    ```
    model = create_model("config.json")
    model.export()
    model.compile()
    ```
    """
    register_all_models()
    if isinstance(config, Mapping):
        config = ModelConfig.create(**config)
    elif not isinstance(config, ModelConfig):
        config = ModelConfig.load(config)

    return config.model_type.from_config(config)


model_registry: dict[str, type["BaseLayer"]] = {}
"""Registry of all model types.
This is used to dispatch when construction a model form a config."""


def register_all_models():
    from .. import models


class BaseLayerMetaClass(ABCMeta):
    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        model_registry[get_model_type_id(cls)] = cls


class BaseLayer(nn.Module, metaclass=BaseLayerMetaClass):
    """Base class of all of our layers."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self._trace_tensor_key_prefix = ""
        self.config = config

        # Can be overridden is derived classes.
        self.default_export_function = "forward"
        self.default_export_batch_sizes = [1]

    @classmethod
    def from_config(cls, config: ModelConfig, /) -> "BaseLayer":
        """Create a model from config.
        Override in derived classes if any special behavior is desired."""
        return cls(config=config)

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        """Return the type of the config for this model."""
        raise NotImplementedError()

    def set_recursively_submodules_default_trace_tensor_key_prefix(self):
        """All submodules get a trace key prefix that reflects their nesting with
        respect to the parent modules.

        Example:
        ```
        class A(BaseLayer):
            def __init__(self):
                ...
                self.b = ...

        class B(BaseLayer):
            def __init__(self):
                ...
                self.c = ...

        class C(BaseLayer):
            def forward(self, x):
                self.trace_tensor("x", x)


        a = A()
        a.trace_tensor_key_prefix = "top."
        a.set_recursively_submodules_default_trace_tensor_key_prefix()
        ```

        This will result in trace key prefixes
        a -> "top."
        a.b -> "top.b."
        a.b.c -> "top.b.c."

        The trace_tensor method call in C.forward will result in a trace with key
        "top.b.c.x".
        """
        _set_recursively_submodules_default_trace_tensor_key_prefix(
            self, self.trace_tensor_key_prefix
        )

    @property
    def trace_tensor_key_prefix(self) -> str:
        """When tracing with self.trace_tensor all keys will be prefixed by this
        string.
        The default prefix is the empty string."""
        return self._trace_tensor_key_prefix

    @trace_tensor_key_prefix.setter
    def trace_tensor_key_prefix(self, value: str):
        self._trace_tensor_key_prefix = value

    def trace_tensor(
        self,
        key: str,
        tensors: Dict[str, torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    ):
        """Trace tensor(s) prefixed by this module's key prefix.

        You can use `set_recursively_submodules_default_trace_tensor_key_prefix` on
        your top level module to specify a key prefix for it and all its nested
        submodules.

        See:
        amdsharktank.layers.BaseLayer.trace_tensor_key_prefix
        amdsharktank.layers.BaseLayer.set_recursively_submodules_default_trace_tensor_key_prefix
        amdsharktank.ops.trace_tensor
        """
        debugging.trace_tensor(f"{self.trace_tensor_key_prefix}{key}", tensors)

    def assert_not_nan(self, *ts: torch.Tensor):
        """Checks whether tensors have nan values in them.

        Must be enabled via a global switch as this kind of checking is not
        accelerator or compilation friendly.
        """
        if debugging.flags.enable_nan_checks:
            for t in ts:
                if torch.isnan(t).any():
                    raise AssertionError(f"Tensor contains nans! {t}")

    def sample_inputs(
        self, batch_size: int | None = 1, function: Optional[str] = None
    ) -> tuple[tuple[AnyTensor, ...], OrderedDict[str, AnyTensor]]:
        """Return sample inputs that can be used to run the function from the model.
        If function is None then layer is treated as the callable.
        E.g.
        ```
        args, kwargs = model.sample_inputs()
        model(*args, **kwargs)
        ```

        One purpose of this method is to standardize exportation of models to MLIR.
        """
        raise NotImplementedError()

    def dynamic_shapes_for_export(
        self,
        batch_size: int | DynamicBatchSize | None = 1,
        function: Optional[str] = None,
    ) -> Dict[str, Any] | tuple[Any, ...] | list[Any] | None:
        """During export the result is directly passed to the underlying export function."""
        return None

    def export_mlir(self, path: PathLike | None = None, /):
        """Export the model into MLIR format.
        Exporting is driven by the model's configuration.
        Can be overridden in derived classes."""

        if path is None:
            path = self.config.mlir_path
        if path is None:
            raise ValueError("Missing MLIR export path.")

        function_batch_sizes_map = self._get_function_batch_sizes_map()
        from amdsharktank.utils.export import export_model_mlir

        export_model_mlir(
            model=self,
            output_path=path,
            function_batch_sizes_map=function_batch_sizes_map,
        )

    def export(self, mlir_path: PathLike | None = None, /, *args, **kwargs):
        """Export MLIR and any other artifacts required for compilation.
        Can be overridden in derived classes."""
        if self.config.export_sample_inputs_enabled:
            path_prefix = mlir_path
            if path_prefix is not None:
                path_prefix = Path(path_prefix)
                path_prefix = path_prefix.parent / path_prefix.stem
            self.export_sample_inputs()

        self.export_mlir(mlir_path)

    def export_sample_inputs(self, path_prefix: PathLike | None = None):
        if path_prefix is None:
            path_prefix = self.config.mlir_path.parent / self.config.mlir_path.stem
        if path_prefix is None:
            raise ValueError("Can't export sample inputs. No path prefix specified.")
        path_prefix = Path(path_prefix)

        function_batch_sizes_map = self._get_function_batch_sizes_map()

        with chdir(str(path_prefix.parent)):
            for function, batch_sizes in function_batch_sizes_map.items():
                for batch_size in batch_sizes:
                    sample_args, sample_kwargs = self.sample_inputs(
                        function=function, batch_size=batch_size
                    )
                    flat_args = flatten_for_iree_signature((sample_args, sample_kwargs))
                    file_path_prefix = (
                        f"{path_prefix.name}-{function}_bs{batch_size}-arg"
                    )
                    arg_descriptors = iree_tool_prepare_input_args(
                        flat_args, file_path_prefix=file_path_prefix
                    )
                    arg_descriptor_path = f"{file_path_prefix}-desc"
                    with open(arg_descriptor_path, "w") as f:
                        for desc in arg_descriptors:
                            print(desc, file=f)

    def compile(self, output_path: PathLike | None = None, /):
        """Compile the model.
        Does not do auto-export, requires the model to be exported first."""
        if output_path is None:
            output_path = self.config.iree_module_path
        if output_path is None:
            raise ValueError("Missing compile output path.")

        self._save_compile_command(Path(output_path))

        from iree.compiler import compile_file

        compile_file(
            str(self.config.mlir_path),
            output_file=str(output_path),
            extra_args=self.config.get_compile_args(),
        )

    def _get_function_batch_sizes_map(self) -> dict[str, list[int]]:
        export_functions = [
            ExportFunctionConfig(
                function=self.default_export_function,
                batch_sizes=self.default_export_batch_sizes,
            )
        ]
        if self.config.export_functions is not None:
            export_functions = self.config.export_functions
        return {
            export_function.function
            or self.default_export_function: export_function.batch_sizes
            or self.default_export_batch_sizes
            for export_function in export_functions
        }

    def _save_compile_command(self, module_path: Path):
        from iree.compiler.tools.core import build_compile_command_line
        from iree.compiler import TempFileSaver, CompilerOptions

        with TempFileSaver.implicit() as tfs:
            options = CompilerOptions(
                output_file=str(module_path), extra_args=self.config.get_compile_args()
            )
            compile_command = build_compile_command_line(
                input_file=str(self.config.mlir_path), tfs=tfs, options=options
            )
        compile_command_file = (
            module_path.parent / f"{module_path.stem}-compile-command"
        )
        with open(compile_command_file, "w") as f:
            f.write(compile_command[0])
            for arg in compile_command[1:]:
                f.write(f" {arg}")
            f.write("\n")


class ThetaLayer(BaseLayer):
    "Base class for layers that derive parameters from a Theta object."

    def __init__(self, theta: Theta | None = None, config: ModelConfig | None = None):
        super().__init__(config=config)
        if theta is None:
            theta = self.load_theta()
        if theta is None:
            theta = self.generate_random_theta()
        self.theta = theta

    def theta_tensor(self, name: str) -> InferenceTensor:
        # TODO: We may need to do some bookkeeping here to ensure export
        # tracks all of these.
        return self.theta.tensor(name)

    def shard_theta(self, theta: Theta) -> Theta:
        """Override to implement theta sharding.
        This default implementation supports only the trivial case of no sharding."""
        if (
            self.config.tensor_parallelism is not None
            and self.config.tensor_parallelism != 1
        ):
            raise ValueError(
                "Theta sharding for model "
                f"{get_model_type_id(self.__class__)} is not supported."
            )
        return theta

    def load_theta(self) -> Theta | None:
        """Load a theta if it exists.
        This can be either an IRPA/GGUF parameters file or a hugging face model."""
        assert self.config is not None

        needs_sharding = True

        parameters_path = self.config.parameters_path
        if parameters_path is not None and parameters_path.exists():
            dataset = Dataset.load(parameters_path)
            theta = dataset.root_theta
            tensor_parallelism = dataset.properties.get("tensor_parallelism", 1)
            if (
                tensor_parallelism != 1
                and tensor_parallelism != self.config.tensor_parallelism
            ):
                raise ValueError(
                    "Could not shard theta that is already sharded "
                    "with different tensor_parallelism. "
                    f"Desired is {self.config.tensor_parallelism}, "
                    f"actual is {tensor_parallelism}"
                )
            needs_sharding = tensor_parallelism != self.config.tensor_parallelism
        elif self.config.hugging_face_repo_id is not None:
            theta = self.load_theta_from_hugging_face()
        else:
            return None

        if needs_sharding:
            theta = self.shard_theta(theta)

        return theta

    def load_theta_from_hugging_face(self) -> Theta:
        """Override to load a theta form Hugging Face."""
        raise NotImplementedError()

    def generate_random_theta(self) -> Theta:
        """Initialize a theta with random contents.
        The generation should respect the model configuration like rng_seed.
        Override in derived classes."""
        raise NotImplementedError()

    def export_parameters(self, path: PathLike | None = None, /):
        "Export model parameters (includes the theta) into an IRPA/GGUF file."
        if path is None:
            path = self.config.export_parameters_path
        if path is None:
            raise ValueError("Missing model parameters export path.")

        properties = self.config.asdict_for_saving()
        dataset = Dataset(properties=properties, root_theta=self.theta)
        dataset.save(path)

    def export(
        self,
        mlir_path: PathLike | None = None,
        parameters_path: PathLike | None = None,
        /,
        *args,
        **kwargs,
    ):
        super().export(mlir_path)
        if (
            parameters_path is not None
            or self.config.export_parameters_path is not None
        ):
            self.export_parameters(parameters_path)
