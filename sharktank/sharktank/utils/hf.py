# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from os import PathLike
import os
import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from sharktank.types import *

logger = logging.getLogger(__name__)


def import_hf_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: Optional[PathLike] = None,
    target_dtype=None,
) -> Optional[Dataset]:
    import safetensors

    with open(config_json_path, "rb") as f:
        config_json = json.load(f)
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json.items() if not k.startswith("_")}

    tensors = []
    for params_path in param_paths:
        with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
            tensors += [
                DefaultPrimitiveTensor(
                    name=name, data=st.get_tensor(name).to(target_dtype)
                )
                for name in st.keys()
            ]

    theta = Theta(tensors)
    props = {
        "meta": meta_params,
        "hparams": hparams,
    }
    dataset = Dataset(props, theta)

    if output_irpa_file is None:
        return dataset

    dataset.save(output_irpa_file, io_report_callback=logger.debug)


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
