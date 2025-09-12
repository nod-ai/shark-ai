# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, TYPE_CHECKING
from os import PathLike
import os
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from sharktank.types import *
from sharktank.utils.functools import compose
from sharktank.transforms.dataset import wrap_in_list_if_inference_tensor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sharktank.types.theta import InferenceTensorTransform

MetadataTransform = Callable[[dict[str, Any]], dict[str, Any]]


def default_metadata_transform(metadata: dict[str, Any]) -> dict[str, Any]:
    meta_params = {k: v for k, v in metadata.items() if k.startswith("_")}
    hparams = {k: v for k, v in metadata.items() if not k.startswith("_")}
    return {
        "meta": meta_params,
        "hparams": hparams,
    }


def import_hf_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: Optional[PathLike] = None,
    target_dtype=None,
    tensor_transform: InferenceTensorTransform | None = None,
    metadata_transform: MetadataTransform | None = None,
) -> Optional[Dataset]:
    import safetensors

    if tensor_transform is None:
        tensor_transform = lambda x: x
    tensor_transform = compose(tensor_transform, wrap_in_list_if_inference_tensor)

    if metadata_transform is None:
        metadata_transform = default_metadata_transform

    tensors = []
    for params_path in param_paths:
        with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
            for name in st.keys():
                tensor = DefaultPrimitiveTensor(
                    name=name, data=st.get_tensor(name).to(target_dtype)
                )
                transformed_tensors = tensor_transform(tensor)
                if transformed_tensors is None:
                    continue
                tensors.extend(transformed_tensors)

    theta = Theta(tensors)

    with open(config_json_path, "rb") as f:
        config_json = json.load(f)
    props = metadata_transform(config_json)

    dataset = Dataset(props, theta)
    if output_irpa_file is not None:
        dataset.save(output_irpa_file, io_report_callback=logger.debug)
    return dataset


def import_hf_dataset_from_hub(
    repo_id: str,
    revision: str | None = None,
    subfolder: str | None = None,
    config_subpath: str | None = None,
    output_irpa_file: PathLike | None = None,
) -> Dataset | None:
    model_dir = Path(snapshot_download(repo_id=repo_id, revision=revision))
    if subfolder is not None:
        model_dir /= subfolder
    if config_subpath is None:
        config_json_path = model_dir / "config.json"
    else:
        config_json_path = model_dir / config_subpath
    file_paths = [
        model_dir / file_name
        for file_name in os.listdir(model_dir)
        if (model_dir / file_name).is_file()
    ]
    param_paths = [p for p in file_paths if p.is_file() and p.suffix == ".safetensors"]

    return import_hf_dataset(
        config_json_path=config_json_path,
        param_paths=param_paths,
        output_irpa_file=output_irpa_file,
    )


def import_hf_llm_dataset(
    repo_id_or_path: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    config_subpath: str | None = None,
    output_irpa_file: PathLike | None = None,
) -> Dataset | None:
    model_dir = Path(repo_id_or_path)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(repo_id=repo_id_or_path, revision=revision))

    if subfolder is not None:
        model_dir /= subfolder
    if config_subpath is None:
        config_json_path = model_dir / "config.json"
    else:
        config_json_path = model_dir / config_subpath
    file_paths = [
        model_dir / file_name
        for file_name in os.listdir(model_dir)
        if (model_dir / file_name).is_file()
    ]
    param_paths = [p for p in file_paths if p.is_file() and p.suffix == ".safetensors"]

    return convert_hf_llm_dataset(
        config_json_path=config_json_path,
        param_paths=param_paths,
        output_irpa_file=output_irpa_file,
    )


def convert_hf_llm_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: PathLike,
):
    pass
