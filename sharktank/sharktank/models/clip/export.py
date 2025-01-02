# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union
import transformers
from transformers.models.clip.modeling_clip import (
    CLIPAttention as HfCLIPAttention,
    CLIPEncoderLayer as HfCLIPEncoderLayer,
    CLIPEncoder as HfCLIPEncoder,
)
import torch
from os import PathLike

from ...types.theta import Theta, Dataset, torch_module_to_theta
from ...layers.configs import ClipTextConfig
from .clip import ClipTextModel
from iree.turbine.aot import FxProgramsBuilder, export


def hugging_face_clip_attention_to_theta(model: HfCLIPAttention) -> Theta:
    return torch_module_to_theta(model)


def hugging_face_clip_encoder_layer_to_theta(model: HfCLIPEncoder) -> Theta:
    return torch_module_to_theta(model)


def hugging_face_clip_encoder_to_theta(model: HfCLIPEncoderLayer) -> Theta:
    return torch_module_to_theta(model)


def hugging_face_clip_text_model_to_theta(model: transformers.CLIPTextModel) -> Theta:
    return torch_module_to_theta(model)


def hugging_face_clip_text_model_to_dataset(
    model: transformers.CLIPTextModel,
) -> Dataset:
    config = ClipTextConfig.from_hugging_face_clip_text_model_config(model.config)
    properties = config.to_properties()
    theta = hugging_face_clip_text_model_to_theta(model)
    theta.rename_tensors_to_paths()
    return Dataset(properties, theta)


def clip_text_model_to_dataset(model: ClipTextModel) -> Dataset:
    return Dataset(properties=model.config.to_properties(), root_theta=model.theta)


def export_clip_text_model_iree_parameters(model: ClipTextModel, output_path: PathLike):
    dataset = clip_text_model_to_dataset(model)
    dataset.save(output_path)


def export_clip_text_model_dataset_from_hugging_face(
    model_or_name_or_path: Union[PathLike, transformers.CLIPTextModel],
    output_path: PathLike,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(model_or_name_or_path, transformers.CLIPTextModel):
        assert dtype is None
        model = model_or_name_or_path
    else:
        model = transformers.CLIPTextModel.from_pretrained(
            model_or_name_or_path, torch_dtype=dtype
        )
    dataset = hugging_face_clip_text_model_to_dataset(model)
    dataset.save(output_path)


def export_clip_text_model_mlir(
    model: Union[ClipTextModel, PathLike],
    batch_sizes: list[int],
    mlir_output_path: str,
):
    """
    Args:
      model: either the torch module or path to GGUF/IRPA.
    """
    if not isinstance(model, ClipTextModel):
        dataset = Dataset.load(model)
        config = ClipTextConfig.from_properties(dataset.properties)
        model = ClipTextModel(theta=dataset.root_theta, config=config)

    fxb = FxProgramsBuilder(model)

    for batch_size in batch_sizes:
        sample_inputs = model.sample_inputs(batch_size)

        @fxb.export_program(
            name=f"forward_bs{batch_size}",
            args=tuple(sample_inputs.values()),
            dynamic_shapes=None,
            strict=False,
        )
        def _(
            model,
            input_ids,
        ):
            return model(input_ids)

    output = export(fxb, import_symbolic_shape_expressions=True)
    output.save_mlir(mlir_output_path)


def export_clip_text_model_to_iree(
    model: ClipTextModel,
    batch_sizes: list[int],
    mlir_output_path: PathLike,
    parameters_output_path: PathLike,
):
    export_clip_text_model_iree_parameters(model, parameters_output_path)
    export_clip_text_model_mlir(
        model=parameters_output_path,
        batch_sizes=batch_sizes,
        mlir_output_path=mlir_output_path,
    )
