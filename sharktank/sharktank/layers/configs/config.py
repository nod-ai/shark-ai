# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, TYPE_CHECKING, ClassVar
import os
from os import PathLike
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from ...types.theta import load_properties
from ...utils import tree, parse_version

if TYPE_CHECKING:
    from ..base import BaseLayer

__all__ = [
    "ExportFunctionConfig",
    "DynamicBatchSize",
    "ModelConfig",
    "model_config_presets",
    "register_model_config_preset",
]


class DynamicBatchSize:
    pass


@dataclass
class ExportFunctionConfig:
    function: str | None = None
    batch_sizes: list[int | DynamicBatchSize | None] | None = None
    """The set of batch sizes to export for.
    The model drives what is the meaning of default values."""


@dataclass(kw_only=True)
class ModelConfig:
    """Base config for common model parameters.
    Specific model configs are meant to inherit this.

    Supports loading and saving from/to json.
    ```
    cfg = ModelConfig.load("/path/to/some.json")
    cfg.save("/path/to/some-other.json")
    ```

    All relative paths in a file config are relative to the config path directory if
    specified. If not specified they are relative to current working directory.
    After the config is loaded relative paths are made relative to CWD.
    """

    current_config_version: ClassVar[str] = "0.1.0"

    model_type: type["BaseLayer"] = None
    """The type of the model that this config is configuring.
    This is used to dispatch to the model to create itself."""

    config_path: Path | None = None
    mlir_path: Path | None = None
    """Export MLIR to this path."""
    iree_module_path: Path | None = None

    parameters_path: Path | None = None
    """Load parameters from this path. IRPA, GGUF, etc."""
    export_parameters_path: Path | None = None
    """Location of parameters when exporting. IRPA, GGUF, etc."""

    hugging_face_repo_id: str | None = None
    hugging_face_revision: str | None = None
    hugging_face_subfolder: str | None = None

    export_functions: list[ExportFunctionConfig] | None = None
    """function, batch size pairs that are to be exported."""
    export_sample_inputs_enabled: bool = False

    tensor_parallelism: int | None = None
    """If specified exported model parameters will be shard."""

    compile_args: list[str] | None = None

    rng_seed: int | None = None
    """Generation of random model weights would use this seed."""

    def __post_init__(self):
        assert self.model_type is not None
        self.model_type = _get_model_type(self.model_type)
        self.config_path = _make_optional_path(self.config_path)
        self.mlir_path = self._config_relative_to_cwd_relative_path(self.mlir_path)
        self.parameters_path = self._config_relative_to_cwd_relative_path(
            self.parameters_path
        )
        self.iree_module_path = self._config_relative_to_cwd_relative_path(
            self.iree_module_path
        )
        if self.export_functions is not None:
            self.export_functions = [
                export_function
                if isinstance(export_function, ExportFunctionConfig)
                else ExportFunctionConfig(**export_function)
                for export_function in self.export_functions
            ]

    @classmethod
    def create(cls, model_type: type["BaseLayer"] | str, **kwargs) -> "ModelConfig":
        """Create a config with type associated with the model_type."""
        model_type_cls = _get_model_type(model_type)
        config_type = model_type_cls.config_type()
        parsed_kwargs = config_type.parse_for_init_kwargs(
            model_type=model_type, **kwargs
        )
        return config_type(**parsed_kwargs)

    @classmethod
    def load(cls, config_path: PathLike, /, **kwargs) -> "ModelConfig":
        """Load a config from json."""
        with open(config_path, "rb") as f:
            config_dict = json.load(f)

        config_dict["config_path"] = config_path
        config_dict.update(kwargs)

        return cls.create(**config_dict)

    def save(self, config_path: PathLike | None = None, /):
        config_path = config_path or self.config_path
        if config_path is None:
            raise ValueError("Could not save config, missing save path")
        with open(config_path, "w") as f:
            json.dump(self.asdict_for_saving(config_path), f)

    def asdict_for_saving(
        self, config_path: PathLike | None = None, /
    ) -> dict[str, Any]:
        """Prepares the parameters to be saved.
        This is meant to be overridden in derived classes where
        special handling of some values is required.
        Here for example Path objects are made relative to the config dir and converted
        to str."""
        config_path = config_path or self.config_path
        config_dir = None
        if config_path is not None:
            config_dir = Path(config_path).parent

        res = self.asdict()

        res["mlir_path"] = self._cwd_relative_to_config_relative_path(
            res["mlir_path"], config_dir
        )
        res["parameters_path"] = self._cwd_relative_to_config_relative_path(
            res["parameters_path"], config_dir
        )
        res["iree_module_path"] = self._cwd_relative_to_config_relative_path(
            res["iree_module_path"], config_dir
        )

        res = {k: v for k, v in res.items() if v is not None}
        if "config_path" in res:
            del res["config_path"]

        def map_leaf_fn(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            return x

        res = tree.map_leaves(
            res, f=map_leaf_fn, is_leaf=tree.is_not_tuple_list_or_dict
        )

        res["config_version"] = self.current_config_version

        from ..base import get_model_type_id

        res["model_type"] = get_model_type_id(self.model_type)
        return res

    def asdict(self) -> dict[str, Any]:
        """This will recurse any fields that are dataclasses and convert them."""
        return asdict(self)

    def get_compile_args(self) -> list[str]:
        if self.compile_args is None:
            return []
        return self.compile_args

    @classmethod
    def translate_hugging_face_config_into_init_kwargs(
        cls, /, repo_id: str, revision: str | None = None, subfolder: str | None = None
    ) -> dict[str, Any]:
        """Download and translate Hugging Face config into key-value pairs that can be
        used to initialize a config."""
        from huggingface_hub import hf_hub_download

        resolved_file = hf_hub_download(
            repo_id, "config.json", subfolder=subfolder, revision=revision
        )
        with open(resolved_file, "rb") as f:
            return cls.translate_hugging_face_config_dict_into_init_kwargs(json.load(f))

    @classmethod
    def translate_hugging_face_config_dict_into_init_kwargs(
        cls, properties: dict[str, Any], /
    ) -> dict[str, Any]:
        """Translate Hugging Face config into key-value pairs that can be used to
        initialize a config."""
        raise NotImplementedError()

    @classmethod
    def parse_for_init_kwargs(cls, **config_dict) -> dict[str, Any]:
        """Prepare arguments for initialization.
        Override in derived classes."""

        cls._check_config_version(config_dict)
        config_dict.pop("config_version")

        config_path = config_dict.get("config_path", ".")

        parameters_path = config_dict.get("parameters_path")
        hugging_face_repo_id = config_dict.get("hugging_face_repo_id")
        if parameters_path is not None and hugging_face_repo_id is not None:
            raise ValueError(
                f"Config values parameters_path and hugging_face_repo_id are mutually exclusive"
            )

        if parameters_path is not None:
            config_dict_from_parameters_file = load_properties(
                Path(config_path) / config_dict["parameters_path"]
            )
            config_dict_from_parameters_file.update(config_dict)
            config_dict = config_dict_from_parameters_file
            if "SHARK_DATASET_VERSION" in config_dict:
                config_dict.pop("SHARK_DATASET_VERSION")

        if hugging_face_repo_id is not None:
            config_form_hf = cls.translate_hugging_face_config_into_init_kwargs(
                config_dict["hugging_face_repo_id"],
                revision=config_dict.get("hugging_face_revision", None),
                subfolder=config_dict.get("hugging_face_subfolder", None),
            )
            config_form_hf.update(config_dict)
            config_dict = config_form_hf

        return config_dict

    @classmethod
    def _check_config_version(cls, config_dict: dict[str, Any], /):
        """Check config version such that the config can be parsed.
        This base class method checks config_version.
        Specific model configs can override this and have other version filed(s) to
        distinguish between versions."""

        version = config_dict.get("config_version")
        if version is None:
            raise ValueError("Missing model config version.")
        if parse_version(version) != parse_version(cls.current_config_version):
            raise ValueError(
                f"Could not load config with a version {version},"
                f"expected version is {parse_version(cls.current_config_version)}"
            )

    def _config_relative_to_cwd_relative_path(
        self, path: Path | str | None
    ) -> Path | None:
        if path is None:
            return path
        path = Path(path)
        if path.is_absolute() or self.config_path is None:
            return path
        return Path(os.path.normpath(self.config_path.parent / path))

    def _cwd_relative_to_config_relative_path(
        self, path: Path | None, config_dir: Path | None = None
    ) -> Path | None:
        if path is None or path.is_absolute() or config_dir is None:
            return path
        return Path(os.path.normpath(os.path.relpath(path, config_dir)))


model_config_presets: dict[str, ModelConfig] = {}
"""Presets of named model configurations."""


def register_model_config_preset(name: str, config: ModelConfig):
    if name in model_config_presets:
        raise ValueError(f'Model config preset with name "{name}" already registered.')
    model_config_presets[name] = config


def _make_optional_path(path: PathLike | None = None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _get_model_type(
    model_type: str | type["BaseLayer"],
) -> type["BaseLayer"]:
    if not isinstance(model_type, str):
        return model_type
    from ..base import model_registry

    return model_registry[model_type]
