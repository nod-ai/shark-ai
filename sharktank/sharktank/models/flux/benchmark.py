# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import iree.compiler
import iree.runtime
import os
from iree.turbine.support.tools import iree_tool_prepare_input_args

from .export import (
    export_flux_transformer_from_hugging_face,
    flux_transformer_default_batch_sizes,
    iree_compile_flags,
)
from ...types import Dataset
from .flux import FluxModelV1, FluxParams
from ...utils.export_artifacts import ExportArtifacts
from ...utils.iree import flatten_for_iree_signature
from ...utils.benchmark import iree_benchmark_module


def iree_benchmark_flux_dev_transformer(
    artifacts_dir: Path,
    iree_device: str,
    json_result_output_path: Path,
    caching: bool = False,
) -> str:
    mlir_path = artifacts_dir / "model.mlir"
    parameters_path = artifacts_dir / "parameters.irpa"
    if (
        not caching
        or not os.path.exists(mlir_path)
        or not os.path.exists(parameters_path)
    ):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
            mlir_output_path=mlir_path,
            parameters_output_path=parameters_path,
        )
    return iree_benchmark_flux_transformer(
        mlir_path=mlir_path,
        parameters_path=parameters_path,
        artifacts_dir=artifacts_dir,
        iree_device=iree_device,
        json_result_output_path=json_result_output_path,
        caching=caching,
    )


def iree_benchmark_flux_transformer(
    artifacts_dir: Path,
    mlir_path: Path,
    parameters_path: Path,
    iree_device: str,
    json_result_output_path: Path,
    caching: bool = False,
) -> str:
    dataset = Dataset.load(parameters_path)
    model = FluxModelV1(
        theta=dataset.root_theta,
        params=FluxParams.from_hugging_face_properties(dataset.properties),
    )
    input_args = flatten_for_iree_signature(
        model.sample_inputs(batch_size=flux_transformer_default_batch_sizes[0])
    )
    cli_input_args = iree_tool_prepare_input_args(
        input_args, file_path_prefix=f"{artifacts_dir / 'arg'}"
    )
    cli_input_args = [f"--input={v}" for v in cli_input_args]

    iree_module_path = artifacts_dir / "model.vmfb"
    if not caching or not os.path.exists(iree_module_path):
        iree.compiler.compile_file(
            mlir_path,
            output_file=iree_module_path,
            extra_args=iree_compile_flags,
        )

    iree_benchmark_args = [
        f"--device={iree_device}",
        f"--module={iree_module_path}",
        f"--parameters=model={parameters_path}",
        f"--function=forward_bs{flux_transformer_default_batch_sizes[0]}",
        "--benchmark_repetitions=30",
        "--benchmark_min_warmup_time=1.0",
        "--benchmark_out_format=json",
        f"--benchmark_out={json_result_output_path}",
    ] + cli_input_args
    return iree_benchmark_module(iree_benchmark_args)
