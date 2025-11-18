# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union
from os import PathLike
import transformers
from transformers.models.clip.modeling_clip import (
    CLIPAttention as HfCLIPAttention,
    CLIPEncoderLayer as HfCLIPEncoderLayer,
    CLIPEncoder as HfCLIPEncoder,
)
import torch

from amdsharktank.types.theta import Theta, Dataset, torch_module_to_theta
from amdsharktank.layers.configs import ClipTextConfig


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
    properties = config.asdict_for_saving()
    theta = hugging_face_clip_text_model_to_theta(model)
    return Dataset(properties, theta)


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
