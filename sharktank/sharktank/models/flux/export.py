# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from os import PathLike
import os
from pathlib import Path
import torch

from ...export import export_model_mlir
from ...tools.import_hf_dataset import import_hf_dataset
from .flux import FluxModelV1, FluxParams
from ...types import Dataset
from ...utils.hf_datasets import get_dataset
from sharktank.transforms.dataset import set_float_dtype
from iree.turbine.aot import (
    ExternalTensorTrait,
)

flux_transformer_default_batch_sizes = [1]


def export_flux_transformer_model_mlir(
    model_or_parameters_path: FluxModelV1 | PathLike,
    /,
    output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    if isinstance(model_or_parameters_path, (PathLike, str)):
        dataset = Dataset.load(model_or_parameters_path)
        model = FluxModelV1(
            theta=dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(dataset.properties),
        )
    else:
        model = model_or_parameters_path

    for t in model.theta.flatten().values():
        ExternalTensorTrait(external_name=t.name, external_scope="").set(t.as_torch())
    export_model_mlir(model, output_path=output_path, batch_sizes=batch_sizes)


def export_flux_transformer_iree_parameters(
    model: FluxModelV1, parameters_output_path: PathLike, dtype=None
):
    model.theta.rename_tensors_to_paths()
    dataset = Dataset(
        root_theta=model.theta, properties=model.params.to_hugging_face_properties()
    )
    if dtype:
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
    dataset.save(parameters_output_path)


def export_flux_transformer(
    model: FluxModelV1,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    export_flux_transformer_iree_parameters(model, parameters_output_path)
    export_flux_transformer_model_mlir(
        model, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def import_flux_transformer_dataset_from_hugging_face(
    repo_id: str, parameters_output_path: PathLike
):
    hf_dataset = get_dataset(
        repo_id,
    ).download()

    import_hf_dataset(
        config_json_path=hf_dataset["config"][0],
        param_paths=hf_dataset["parameters"],
        output_irpa_file=parameters_output_path,
    )


def export_flux_transformer_from_hugging_face(
    repo_id: str,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    import_flux_transformer_dataset_from_hugging_face(
        repo_id=repo_id, parameters_output_path=parameters_output_path
    )
    export_flux_transformer_model_mlir(
        parameters_output_path, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def export_flux_transformer_models(dir: Path):
    from .testing import export_dev_random_single_layer

    base_dir = dir / "flux" / "transformer"
    os.makedirs(base_dir)

    file_name_base = "black-forest-labs--FLUX.1-dev--black-forest-labs-transformer-bf16"
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_flux_transformer_from_hugging_face(
        "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )

    file_name_base = (
        "black-forest-labs--FLUX.1-schnell--black-forest-labs-transformer-bf16"
    )
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_flux_transformer_from_hugging_face(
        "black-forest-labs/FLUX.1-schnell/black-forest-labs-transformer",
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )

    file_name_base = "black-forest-labs--FLUX.1-dev--transformer-single-layer-b16"
    mlir_path = base_dir / f"{file_name_base}.mlir"
    parameters_output_path = base_dir / f"{file_name_base}.irpa"
    export_dev_random_single_layer(
        dtype=torch.bfloat16,
        mlir_output_path=mlir_path,
        parameters_output_path=parameters_output_path,
    )
