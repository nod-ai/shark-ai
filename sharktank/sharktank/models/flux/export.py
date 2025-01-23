# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import PathLike
import os
from pathlib import Path
import torch

from ...export import export_static_model_mlir
from ...tools.import_hf_dataset import import_hf_dataset
from .flux import FluxModelV1, FluxParams
from ...types import Dataset
from ...utils.hf_datasets import get_dataset

flux_transformer_default_batch_sizes = [1]

iree_compile_flags = [
    "--iree-hal-target-device=hip",
    "--iree-hip-target=gfx942",
    "--iree-opt-const-eval=false",
    "--iree-opt-strip-assertions=true",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-opt-data-tiling=false",
    "--iree-codegen-gpu-native-math-precision=true",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-hip-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
]


def export_flux_transformer_model_mlir(
    model: FluxModelV1,
    output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    export_static_model_mlir(model, output_path=output_path, batch_sizes=batch_sizes)


def export_flux_transformer_iree_parameters(
    model: FluxModelV1, parameters_output_path: PathLike
):
    model.theta.rename_tensors_to_paths()
    # TODO: export properties
    dataset = Dataset(root_theta=model.theta, properties={})
    dataset.save(parameters_output_path)


def export_flux_transformer(
    model: FluxModelV1,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = flux_transformer_default_batch_sizes,
):
    export_flux_transformer_iree_parameters(model, parameters_output_path)

    dataset = Dataset.load(parameters_output_path)
    model_with_frozen_theta = FluxModelV1(theta=dataset.root_theta, params=model.params)
    model_with_frozen_theta.theta = dataset.root_theta
    export_flux_transformer_model_mlir(
        model_with_frozen_theta, output_path=mlir_output_path, batch_sizes=batch_sizes
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

    dataset = Dataset.load(parameters_output_path)
    model = FluxModelV1(
        theta=dataset.root_theta,
        params=FluxParams.from_hugging_face_properties(dataset.properties),
    )
    export_flux_transformer_model_mlir(
        model, output_path=mlir_output_path, batch_sizes=batch_sizes
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
