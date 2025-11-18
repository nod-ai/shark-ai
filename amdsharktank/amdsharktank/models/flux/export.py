# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import os
from os import PathLike
from pathlib import Path

from amdsharktank.types import Dataset
from amdsharktank.layers import create_model, model_config_presets
from amdsharktank.transforms.dataset import set_float_dtype
from amdsharktank.utils import chdir
from amdsharktank.utils.export import export_model_mlir
from amdsharktank.utils.iree import trace_model_with_tracy
from amdsharktank.utils.hf import import_hf_dataset_from_hub
from .flux import FluxModelV1, FluxParams

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
    repo_id: str,
    revision: str | None = None,
    subfolder: str | None = None,
    parameters_output_path: PathLike | None = None,
) -> Dataset | None:
    return import_hf_dataset_from_hub(
        repo_id,
        revision=revision,
        subfolder=subfolder,
        config_subpath="transformer/config.json",
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
    variants = ["dev", "schnell"]
    iree_hal_target_device = "hip"
    iree_hip_target = "gfx942"
    output_img_height = 1024
    output_img_width = 1024
    build_types = ["debug", "release"]

    preset_model_names = [
        f"black-forest-labs--FLUX.1-{variant}-bf16-{output_img_height}x{output_img_width}-{iree_hal_target_device}-{iree_hip_target}-{build_type}"
        for variant in variants
        for build_type in build_types
    ]

    base_dir = dir / "flux" / "transformer"
    os.makedirs(base_dir, exist_ok=True)
    for variant in variants:
        for build_type in build_types:
            model_name = f"black-forest-labs--FLUX.1-{variant}-bf16-{output_img_height}x{output_img_width}-{iree_hal_target_device}-{iree_hip_target}-{build_type}"
            with chdir(base_dir):
                model = create_model(model_config_presets[model_name])
                model.export()
                model.compile()
                if build_type == "debug":
                    trace_model_with_tracy(
                        model.config,
                        function="forward_bs1",
                        output_trace_path=f"{model.config.iree_module_path}.tracy",
                    )
