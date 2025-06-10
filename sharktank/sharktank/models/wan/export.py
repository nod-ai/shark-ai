# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from os import PathLike
from pathlib import Path
from typing import Any, Optional
from collections import OrderedDict
import math

from sharktank.types import Dataset, AnyTensor
from sharktank.layers import create_model, model_config_presets, ThetaLayer
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils import chdir
from sharktank.utils.export import export_model_mlir
from sharktank.utils.iree import trace_model_with_tracy
from sharktank.utils.hf import import_hf_dataset_from_hub
from .wan import WanModel, WanParams

from iree.turbine.aot import (
    ExternalTensorTrait,
)

import torch
import torch.nn as nn

wan_transformer_default_batch_sizes = [1]

class WanTransformerWrapped(ThetaLayer):
    def __init__(self, model, num_frames = 81, height = 720, width = 1280):
        super().__init__(
            config=model.config,
            theta=model.theta,
        )
        self.model = model
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.vae_stride = (4, 8, 8)
        self.vae_z_dim = 16

    def forward(self, x, t, context):
        x = [x.type(torch.bfloat16)]
        context = [context.type(torch.bfloat16)]
        res = self.model.forward(x, t, context)
        return [r.type(torch.float16) for r in res]
    
    def _get_noise(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ):
        F = num_frames
        return [torch.randn(
            self.vae_z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            # allow for packing
            height // self.vae_stride[1],
            width // self.vae_stride[2],
            dtype=dtype,
        )] * batch_size

    def sample_inputs(
        self, batch_size: int = 1, function: Optional[str] = None
    ) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:
        if not (function is None or function == "forward_t2v"):
            raise ValueError(f'Only function "forward_t2v" is supported. Got "{function}"')
        
        # Prepare inputs
        # input config
      
        # Get wan model input
        model_input = self._get_noise(
            batch_size,
            self.num_frames,
            self.height,
            self.width,
        )
        if function == "forward_t2v":
            context_shape = (28, 4096)
            self.model.set_export_config(seq_len=75600)
            args = tuple()
            kwargs = OrderedDict(
                (
                    ("x", model_input[0]),
                    ("t", torch.tensor([999], dtype=torch.int64)),
                    ("context", torch.rand(context_shape, dtype=torch.float16)),
                )
            )
        # else:
        #     args = tuple()
        #     self.set_export_config(seq_len=max_seq_len)
        #     kwargs = OrderedDict(
        #         (
        #             ("x", [model_input]),
        #             ("t", timestep_input),
        #             ("context", [text_embeddings[0]]),
        #             ("clip_fea", clip_image_embeddings[:1]),
        #             ("y", [ys]),
        #         )
        #     )
        return args, kwargs


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
    fn_bs_map = {
        "forward_t2v": [*batch_sizes]
    }
    wrapped_model = WanTransformerWrapped(model)
    # golden = wrapped_model.forward(**wrapped_model.sample_inputs(function="forward_t2v")[1])
    export_model_mlir(wrapped_model, output_path=output_path, function_batch_sizes_map=fn_bs_map)


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
