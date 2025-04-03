# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Union, Collection, Sequence, List
from typing_extensions import deprecated
from tempfile import TemporaryFile
import json
from pathlib import Path
from dataclasses import dataclass
import warnings
from os import PathLike

import torch

from iree.turbine.aot import (
    ExternalTensorTrait,
    ParameterArchive,
    ParameterArchiveEntry,
)

from sharktank.utils.io import ShardedArchiveBuilder

from .tensors import (
    InferenceTensor,
    InferenceTensorMetadata,
    DefaultPrimitiveTensor,
    REGISTERED_INFERENCE_TENSOR_CLASSES,
    unbox_tensor,
)

__all__ = [
    "Dataset",
    "flat_to_nested_dict",
    "load_properties",
    "Theta",
    "torch_module_to_theta",
    "mark_export_external_theta",
    "save_load_theta",
]

IOReportCallback = Callable[[str], None]


################################################################################
# Theta object
# A theta object represents a hierarchical pack of parameters. All parameters
# are InferenceTensor objects, meaning that they can either be raw PyTorch
# tensors or composite/packed QuantizedTensors.
#
# As in classic implementations, we separate the theta parameter pack from the
# model code because there are many interesting transformations that can be
# done on it in isolation.
################################################################################

InferenceTensorTransform = Callable[
    [InferenceTensor], Union[None, InferenceTensor, Sequence[InferenceTensor]]
]


class InferenceTensorTransforms:
    """Container for common transformations on InferenceTensors."""

    @staticmethod
    def identity() -> InferenceTensorTransform:
        return lambda x: x

    @staticmethod
    def to_device(
        device: Optional[Union[str, torch.device]]
    ) -> InferenceTensorTransform:
        if device is not None:
            return lambda it: it.to(device=device)
        return InferenceTensorTransforms.identity()


class Theta:
    """Subset of parameter tensors used for inference."""

    def __init__(
        self,
        tensors: Union[Sequence[InferenceTensor], dict[str, dict | InferenceTensor]],
    ):
        if not isinstance(tensors, dict):
            tensors = {t.name: t for t in tensors}
        assert all(isinstance(k, str) for k in _all_keys(tensors))
        assert all(
            v is None or isinstance(v, InferenceTensor) for v in _leaf_values(tensors)
        )
        self._tree = flat_to_nested_dict(tensors)

    def transform(self, *transforms: InferenceTensorTransform) -> "Theta":
        """Transforms all inference tensors by applying transform functions.

        Returns a modified theta.
        """
        orig_flat_tensors = self.flatten().values()
        for transform in transforms:
            tran_flat_tensors = []
            for it in orig_flat_tensors:
                results = transform(it)
                if results is None:
                    continue
                if isinstance(results, InferenceTensor):
                    tran_flat_tensors.append(results)
                else:
                    tran_flat_tensors.extend(results)
            orig_flat_tensors = tran_flat_tensors

        return Theta(orig_flat_tensors)

    def to(self, *, device: Optional[Union[str, torch.device]] = None) -> "Theta":
        return self.transform(InferenceTensorTransforms.to_device(device))

    def pop(self, *name_path: str | int) -> "Theta":
        # prune a subtree from the tree and return it as a new Theta object
        name_path = ".".join(_norm_name_path(name_path))
        flat = self.flatten()
        accum = {}
        key_list = list(flat.keys())
        for key in key_list:
            if key.startswith(name_path):
                accum[key] = flat.pop(key)
        self._tree = flat_to_nested_dict(flat)
        return Theta(flat_to_nested_dict(accum))

    def flatten(self) -> dict[str, InferenceTensor]:
        results = {}

        def accum(prefix, child):
            for key, value in child.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    accum(new_prefix, value)
                else:
                    results[new_prefix] = value

        accum("", self._tree)
        return results

    def tensor(self, *name_path: str | int) -> InferenceTensor:
        name_path = _norm_name_path(name_path)
        t = self.optional_tensor(*name_path)
        if t is None:
            raise KeyError(
                f"Could not find tensor {name_path[-1]} in theta {name_path[0:-1]}"
            )
        return t

    def optional_tensor(self, *name_path: str | int) -> Optional[InferenceTensor]:
        name_path = _norm_name_path(name_path)
        try:
            current_ts = self._tree
            for part in name_path[0:-1]:
                current_ts = current_ts[str(part)]
            last = name_path[-1]
        except KeyError:
            raise KeyError(
                f"Unknown parameter {name_path} (in Theta object "
                f"containing {self.keys})"
            )
        t = current_ts.get(str(last))
        return t

    @property
    def keys(self) -> Collection[str]:
        return self._tree.keys()

    def __contains__(self, key) -> bool:
        return key in self._tree

    @property
    def tensors(self) -> Collection[InferenceTensor]:
        return [v for v in self._tree.values() if isinstance(v, InferenceTensor)]

    @property
    def tree(self) -> dict[str, dict | InferenceTensor]:
        """The nested structure of named tensors."""
        return self._tree

    def __call__(self, *name_path: str | int) -> Union["Theta", InferenceTensor]:
        name_path = _norm_name_path(name_path)
        current_ts = self._tree
        try:
            for part in name_path:
                current_ts = current_ts[str(part)]
        except KeyError:
            raise KeyError(f"Sub-theta {name_path} not found (of {self._tree.keys()})")
        if isinstance(current_ts, InferenceTensor):
            return current_ts
        return Theta(current_ts)

    def __repr__(self):
        return f"Theta({self.keys})"

    def add_tensors_to_archive(
        self,
        irpa: "ShardedArchiveBuilder",
        inference_tensor_metas: dict[str, InferenceTensorMetadata],
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        """Adds tensors to the given archive builder."""
        for inference_tensor in self.flatten().values():
            if io_report_callback:
                io_report_callback(f"Add {inference_tensor}")
            name = inference_tensor.name
            if name in inference_tensor_metas:
                warnings.warn(f"Duplicate inference tensor added to archive: {name}")
            meta = inference_tensor.add_to_archive(irpa)
            inference_tensor_metas[name] = meta

    def rename_tensors_to_paths(self):
        """Rename each tensor to have name equal to its path in the theta.
        Example: name="a.b.c"
        """
        for path, tensor in self.flatten().items():
            tensor.name = path


def torch_module_to_theta(module: torch.nn.Module) -> Theta:
    res = Theta(
        {
            name: DefaultPrimitiveTensor(data=param, name=name)
            for name, param in module.named_parameters()
        }
    )
    mark_export_external_theta(res)
    return res


def mark_export_external_theta(theta: Theta):
    """Mark all tensors as external.
    This will cause during export these tensors to not be embedded into the MLIR."""
    for _, tensor in theta.flatten().items():
        ExternalTensorTrait(external_name=tensor.name, external_scope="").set(
            unbox_tensor(tensor)
        )
    return theta


def flat_to_nested_dict(flat: dict[str, Any]) -> dict[str, Any]:
    """Nest a flat or semi-flat dictionary.

    The key nesting separator is the "." symbol.
    Example:
    ```python
    flat_to_nested_dict({
        "a.b": 0,
        "a": { "c": 1 }
    })
    ```

    Results in:
    ```python
    {
        "a": {
            "b": 0,
            "c": 1
        }
    }
    ```
    """
    nested: dict = {}

    def add_to_dict(
        name: str,
        value,
    ):
        current = nested

        parts = name.split(".")
        for part in parts[0:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            assert isinstance(
                current, dict
            ), f"Name collision in parameter dict: {name}"
        if value is not None:
            current[parts[-1]] = value

    for name, value in flat.items():
        add_to_dict(name, value)
    return nested


def _leaf_values(d: dict) -> List[Any]:
    res = []
    for v in d.values():
        if isinstance(v, dict):
            res.extend(_leaf_values(v))
        else:
            res.append(v)
    return res


def _all_keys(d: dict) -> List[Any]:
    res = []
    for k, v in d.items():
        res.append(k)
        if isinstance(v, dict):
            res.extend(_all_keys(v))
    return res


def _norm_name_path(name_parts) -> list[str]:
    accum = []
    for part in name_parts:
        part = str(part)
        accum.extend(part.split("."))
    return accum


@deprecated("Use mark_export_external_theta instead.")
def save_load_theta(theta: Theta) -> Theta:
    """Roundtrip to disk to avoid treating parameters as constants that would appear
    in the MLIR."""
    theta.rename_tensors_to_paths()
    dataset = Dataset(root_theta=theta, properties={})
    with TemporaryFile(prefix="save_load_theta", suffix=".irpa") as file_path:
        dataset.save(file_path)
        return Dataset.load(file_path).root_theta


################################################################################
# Dataset objects
#
# A dataset object combines a root theta and a dictionary of properties
# defining the computation characteristics that the parameters were trained for
# (i.e. hyperparams).
#
# Note that model implementation parameters are represented elsewhere (i.e. for
# things that involve selecting an implementation that meets certain
# characteristics).
################################################################################

PropertyValueType = Union[
    int, float, bool, list["PropertyValueType"], dict[str, "PropertyValueType"]
]


@dataclass
class Dataset:
    """Top level configuration for a model.

    This consists of:

    * Dataset level hyper-parameters (properties).
    * Root theta with materialized parameters (Theta).
    """

    properties: dict[str, PropertyValueType]
    root_theta: Theta

    def save(
        self,
        path: Union[str, Path],
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        """Saves a parameter archive consisting of properties and theta.

        By default, all quantized tensors in theta which do not have a custom
        packed serialization are converted to a generic planar form.

        Sufficient metadata is stored such that `load()` can reconstitute the
        Dataset.
        """
        _dataset_save_helper(self, path, io_report_callback=io_report_callback)

    @staticmethod
    def load(
        path: Union[str, Path],
        *,
        file_type: Optional[str] = None,
        mmap: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Dataset":
        """Loads a dataset from a parameter archive constructed with save."""
        ds = _dataset_load_helper(path, file_type=file_type, mmap=mmap)
        if device is not None:
            ds.to(device=device)
        return ds

    def transform(self, *transforms: InferenceTensorTransform):
        """Does an in-place transformation of `root_theta`.

        The result of the transformation is stored back into `root_theta`.
        """
        self.root_theta = self.root_theta.transform(*transforms)

    def to(self, *, device: Optional[Union[str, torch.device]] = None):
        self.transform(InferenceTensorTransforms.to_device(device))


################################################################################
# Dataset I/O helpers
################################################################################


@dataclass
class DatasetMetadata:
    """Container for serialization state of a dataset.

    When saved to an IRPA file, it will be saved with multiple keys:

    * properties: __SHARK_DATASET__
    * inference_tensors: __SHARK_INFERENCE_TENSORS__
    """

    properties: dict
    inference_tensors: dict[str, InferenceTensorMetadata]
    shard_ranks: tuple[int] = ()

    def save(
        self,
        builder: ShardedArchiveBuilder,
        *,
        io_report_callback: Optional[IOReportCallback] = None,
    ):
        properties_object = self.properties
        properties_object["SHARK_DATASET_VERSION"] = 1
        inference_tensors_object = {
            k: v.to_json() for k, v in self.inference_tensors.items()
        }

        # __SHARK_DATASET__ properties blob.
        try:
            properties_json_blob = json.dumps(properties_object, indent=2)
        except TypeError as e:
            raise TypeError(
                f"Illegal dataset properties object: {properties_object}"
            ) from e
        if io_report_callback:
            import textwrap

            io_report_callback(
                f"Add __SHARK_DATASET__:\n{textwrap.indent(properties_json_blob, '    ')}"
            )
        builder.add_blob("__SHARK_DATASET__", properties_json_blob.encode())

        # __SHARK_SHARD_RANKS__ list.
        if self.shard_ranks:
            shard_ranks_blob = json.dumps(self.shard_ranks)
            if io_report_callback:
                io_report_callback(
                    f"Add __SHARK_SHARD_RANKS__: {shard_ranks_blob.encode()}"
                )
            builder.add_blob("__SHARK_SHARD_RANKS__", shard_ranks_blob.encode())

        # __SHARK_INFERENCE_TENSORS__ blob.
        try:
            inference_tensors_blob = json.dumps(inference_tensors_object, indent=2)
        except TypeError as e:
            raise TypeError(
                f"Illegal inference tensor object: {inference_tensors_object}"
            ) from e
        if io_report_callback:
            import textwrap

            io_report_callback(
                f"Add __SHARK_INFERENCE_TENSORS__:\n{textwrap.indent(inference_tensors_blob, '    ')}\n"
            )
        builder.add_blob("__SHARK_INFERENCE_TENSORS__", inference_tensors_blob.encode())

    def load_metadata(self, entries: dict[str, ParameterArchiveEntry]):
        # Load properties.
        try:
            properties_entry = entries["__SHARK_DATASET__"]
        except KeyError:
            raise IOError(
                f"Parameter archive does not contains __SHARK_DATASET__. Was it produced by this tool?"
            )
        properties_obj = json.loads(bytes(properties_entry.raw.file_view))
        assert isinstance(properties_obj, dict)
        self.properties.update(properties_obj)

        # __SHARK_SHARD_RANKS__
        shard_ranks_entry = entries.get("__SHARK_SHARD_RANKS__")
        if shard_ranks_entry is not None:
            shark_ranks_obj = json.loads(bytes(shard_ranks_entry.raw.file_view))
            assert isinstance(shark_ranks_obj, list) and all(
                isinstance(i, int) for i in shark_ranks_obj
            )
            self.shard_ranks = tuple(shark_ranks_obj)

    def load_tensors(self, entries: dict[str, ParameterArchiveEntry]):
        # Load inference tensors.
        try:
            inference_tensors_entry = entries["__SHARK_INFERENCE_TENSORS__"]
        except KeyError:
            raise IOError(
                f"Parameter archive does not contains __SHARK_INFERENCE_TENSORS__. Was it produced by this tool?"
            )
        inference_tensors_obj = json.loads(bytes(inference_tensors_entry.raw.file_view))
        assert isinstance(inference_tensors_obj, dict)

        inference_tensors = self.inference_tensors
        for tensor_name, tensor_meta_obj in inference_tensors_obj.items():
            tensor_meta = InferenceTensorMetadata.from_json(tensor_meta_obj)
            # Map the raw_tensors dict to tensors from the archive.
            raw_tensors = {}
            for local_name, global_name in tensor_meta.raw_tensors.items():
                try:
                    raw_entry = entries[global_name]
                except KeyError as e:
                    raise IOError(
                        f"InferenceTensor missing one of its tensor components"
                    ) from e
                raw_tensor = raw_entry.as_tensor()
                # Tag the tensor as originating from external storage. This will
                # make any subsequent compilation with it expect to load it from
                # the same parameter archive.
                ExternalTensorTrait(external_name=global_name, external_scope="").set(
                    raw_tensor
                )
                raw_tensors[local_name] = raw_tensor

            # Instantiate the tensor.
            try:
                tensor_clazz = REGISTERED_INFERENCE_TENSOR_CLASSES[
                    tensor_meta.type_name
                ]
            except KeyError as e:
                raise IOError(
                    f"Unregistered InferenceTensor deserialization type"
                ) from e
            inference_tensor = tensor_clazz.create(
                tensor_name, raw_tensors, tensor_meta.extra_properties
            )
            inference_tensors[tensor_name] = inference_tensor


def _dataset_save_helper(
    dataset: Dataset,
    path: Union[str, Path],
    *,
    io_report_callback: Optional[IOReportCallback] = None,
):
    builder = ShardedArchiveBuilder(Path(path))
    ds_meta = DatasetMetadata(dict(dataset.properties), {})
    # Add tensors.
    dataset.root_theta.add_tensors_to_archive(
        builder,
        ds_meta.inference_tensors,
        io_report_callback=io_report_callback,
    )
    ds_meta.shard_ranks = tuple(builder._rank_builders.keys())

    # Add metadata.
    ds_meta.save(builder, io_report_callback=io_report_callback)

    if io_report_callback:
        io_report_callback("Saving file")
    builder.commit()


def _dataset_load_helper(
    path: Union[str, Path],
    *,
    file_type: Optional[str] = None,
    mmap: bool = True,
) -> Dataset:
    path = Path(path)
    suffixes = path.suffixes
    if file_type == "gguf" or suffixes[-1] == ".gguf":
        from . import gguf_interop

        return gguf_interop.load_file(path)
    elif file_type == "irpa" or suffixes[-1] == ".irpa":
        return _dataset_load_irpa(path, mmap=mmap)
    else:
        raise IOError(
            f"Unknown file type '{''.join(path.suffixes)} for loading a Dataset"
        )


def _dataset_load_irpa(path: Path, mmap: bool) -> Dataset:
    # Need to load in two phases: first read metadata from the root archive.
    meta = DatasetMetadata(properties={}, inference_tensors={})
    archive = ParameterArchive(path, mmap=mmap)
    entries = {k: v for k, v in archive.items()}
    meta.load_metadata(entries)

    # Then we know what side-car rank archives should exist, so load those.
    for rank in meta.shard_ranks:
        rank_path = ShardedArchiveBuilder.path_for_rank(path, rank)
        archive.load(rank_path, mmap=mmap)

    # Finally, load all inference tensors.
    entries = {k: v for k, v in archive.items()}
    meta.load_tensors(entries)

    # Note that there may be duplicates. Last wins.
    dataset = Dataset(meta.properties, Theta(meta.inference_tensors))
    return dataset


def load_irpa_properties(path: PathLike, /) -> dict[str, PropertyValueType]:
    meta = DatasetMetadata(properties={}, inference_tensors={})
    archive = ParameterArchive(path)
    entries = {k: v for k, v in archive.items()}
    meta.load_metadata(entries)
    return meta.properties


def load_properties(path: PathLike, /) -> dict[str, Any]:
    path = Path(path)
    suffixes = path.suffixes
    if suffixes[-1] == ".gguf":
        from . import gguf_interop

        return gguf_interop.load_properties(path)
    elif suffixes[-1] == ".irpa":
        return load_irpa_properties(path)
    else:
        # TODO: implement detection of file type from magic number.
        raise ValueError(
            f"Could not load properties form file {path}, unknown file type"
        )
