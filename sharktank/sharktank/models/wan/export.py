# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from os import PathLike
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import functools
import logging

from sharktank.types import Dataset, AnyTensor
from sharktank.layers import create_model, model_config_presets, ThetaLayer
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils import chdir
from sharktank.utils.export import mark_export_external_theta
from sharktank.utils.iree import trace_model_with_tracy
from sharktank.utils.hf import import_hf_dataset_from_hub
from .wan import WanModel, WanConfig

from iree.turbine.aot import *

import numpy as np

import torch

logger = logging.getLogger(__name__)
wan_transformer_default_batch_sizes = [1]


def export_wan_transformer_iree_parameters(
    model: WanModel, parameters_output_path: PathLike, dtype=None
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


def export_wan_transformer_model_mlir(
    model_or_parameters_path: WanModel | PathLike,
    output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
    height: int = 512,
    width: int = 512,
    num_frames: int = 81,
):
    if isinstance(model_or_parameters_path, (PathLike, str)):
        dataset = Dataset.load(model_or_parameters_path)
        for key, value in dataset.properties.items():
            log(f"{key}: {value}")  # SHARK_DATASET_VERSION: 1

        model = WanModel(
            theta=dataset.root_theta,
            params=WanConfig.from_hugging_face_properties(dataset.properties),
        )
    else:
        model = model_or_parameters_path

    model.set_export_config(height=height, width=width, frame_num=num_frames)

    for t in model.theta.flatten().values():
        ExternalTensorTrait(external_name=t.name, external_scope="").set(t.as_torch())

    fn_bs_map = {"forward_t2v": [*batch_sizes]}

    logger.info("Exporting MLIR...")
    export_model_mlir(
        model, output_path=output_path, function_batch_sizes_map=fn_bs_map
    )


def export_wan_transformer(
    model: WanModel,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
):
    export_wan_transformer_iree_parameters(model, parameters_output_path)
    export_wan_transformer_model_mlir(
        model, output_path=mlir_output_path, batch_sizes=batch_sizes
    )


def import_wan_transformer_dataset_from_hugging_face(
    repo_id: str,
    revision: str | None = None,
    subfolder: str | None = None,
    parameters_output_path: PathLike | None = None,
    dtype: str | None = None,
) -> Dataset | None:
    dataset = import_hf_dataset_from_hub(
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
        config_subpath="config.json",
        allow_patterns=["diffusion_pytorch_model*", "config*"],
    )
    if dtype:
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
    dataset.save(parameters_output_path, io_report_callback=logger.debug)
    for key, value in dataset.properties.items():
        logger.debug(f"{key}: {value}")
    return parameters_output_path


def export_wan_transformer_from_hugging_face(
    repo_id: str,
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
    batch_sizes: list[int] = wan_transformer_default_batch_sizes,
    height: int = 512,
    width: int = 512,
    num_frames: int = 81,
    dtype: torch.dtype = torch.bfloat16,
):
    if not os.path.exists(parameters_output_path):
        logger.info(
            f"Wan2.1 transformer IRPA not found. Importing from huggingface ({repo_id})"
        )
        import_wan_transformer_dataset_from_hugging_face(
            repo_id=repo_id, parameters_output_path=parameters_output_path, dtype=dtype
        )
    export_wan_transformer_model_mlir(
        parameters_output_path,
        output_path=mlir_output_path,
        batch_sizes=batch_sizes,
        height=height,
        width=width,
        num_frames=num_frames,
    )


def export_model_mlir(
    model,
    output_path: PathLike,
    *,
    function_batch_sizes_map: Optional[dict[Optional[str], list[int]]] = None,
    batch_sizes: Optional[list[int]] = None,
):
    """Export a model with no dynamic dimensions.

    For the set of provided function name batch sizes pair, the resulting MLIR will
    have function names with the below format.
    ```
    <function_name>_bs<batch_size>
    ```

    If `batch_sizes` is given then it defaults to a single function with named
    "forward".

    The model is required to implement method `sample_inputs`.
    """

    assert not (function_batch_sizes_map is not None and batch_sizes is not None)

    if isinstance(model, ThetaLayer):
        mark_export_external_theta(model.theta)

    if batch_sizes is not None:
        function_batch_sizes_map = {None: batch_sizes}

    if function_batch_sizes_map is None and batch_sizes is None:
        function_batch_sizes_map = {None: batch_sizes}

    decomp_list = [
        torch.ops.aten.logspace,
        torch.ops.aten.upsample_bicubic2d.vec,
        torch.ops.aten._upsample_nearest_exact2d.vec,
        torch.ops.aten.as_strided,
        torch.ops.aten.as_strided_copy.default,
        torch.ops.aten.outer,
    ]
    decomp_blacklist = [
        torch.ops.aten.slice,
        torch.ops.aten.chunk,
        torch.ops.aten.split,
    ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
        remove_ops=decomp_blacklist,
    ):
        fxb = FxProgramsBuilder(model)

        for function, batch_sizes in function_batch_sizes_map.items():
            for batch_size in batch_sizes:
                args, kwargs = model.sample_inputs(batch_size, function)
                dynamic_shapes = model.dynamic_shapes_for_export(
                    batch_size=batch_size, function=function
                )

                @fxb.export_program(
                    name=f"{function or 'forward'}_bs{batch_size}",
                    args=args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
                def _(model, **kwargs):
                    return model(**kwargs)

        output = export(fxb)
        output.save_mlir(output_path)

    logger.info("Saved MLIR to: ", str(output_path))
