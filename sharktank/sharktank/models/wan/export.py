# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from os import PathLike
from pathlib import Path
from sharktank.types import Dataset
from sharktank.layers import create_model, model_config_presets
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils import chdir
from sharktank.utils.export import export_model_mlir
from sharktank.utils.iree import trace_model_with_tracy
from sharktank.utils.hf import import_hf_dataset_from_hub
from .wan import WanModel, WanParams

from iree.turbine.aot import (
    ExternalTensorTrait,
)

wan_transformer_default_batch_sizes = [1]


def export_wan_transformer_model_mlir(
    model_or_parameters_path: WanModel | PathLike,
    output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
):
    if isinstance(model_or_parameters_path, (PathLike, str)):
        dataset = Dataset.load(model_or_parameters_path)
        for key, value in dataset.properties.items():
            print(f"{key}: {value}") # SHARK_DATASET_VERSION: 1

        model = WanModel(
            theta=dataset.root_theta,
            params=WanParams.get_wan_params(),
        )
    else:
        model = model_or_parameters_path

    for t in model.theta.flatten().values():
        ExternalTensorTrait(external_name=t.name, external_scope="").set(t.as_torch())
    export_model_mlir(model, output_path=output_path, batch_sizes=batch_sizes)


def import_wan_transformer_dataset_from_huggingface(
    repo_id: str,
    revision: str | None = None,
    subfolder: str | None = None,
    parameters_output_path: PathLike | None = None,
) -> Dataset | None:
    return import_hf_dataset_from_hub(
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        config_subpath="config.json",
        output_irpa_file=parameters_output_path,
    )


def export_wan_transformer_from_huggingface(
    repo_id: str,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
):
    import_wan_transformer_dataset_from_huggingface(
        repo_id=repo_id, parameters_output_path=parameters_output_path
    )
    export_wan_transformer_model_mlir(
        parameters_output_path, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def export_wan_transformer_models(dir: Path):
    variants = ["t2v", "i2v"]
    iree_hal_target_device = "hip"
    iree_hip_target = "gfx942"
    output_img_height = 512
    output_img_width = 512
    build_types = ["debug", "release"]

    base_dir = dir / "wan" / "transformer"
    os.makedirs(base_dir, exist_ok=True)
    for variant in variants:
        for build_type in build_types:
            model_name = f"wan2.1-{variant}-bf16-{output_img_height}x{output_img_width}-{iree_hal_target_device}-{iree_hip_target}-{build_type}"
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
