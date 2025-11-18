# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import Any, Optional, Union
from pathlib import Path
from copy import copy
import transformers
import torch

from .t5 import T5Config, T5Encoder
from amdsharktank.types import (
    Dataset,
    Theta,
    DefaultPrimitiveTensor,
    torch_module_to_theta,
)
from amdsharktank.transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export, ExternalTensorTrait

__all__ = [
    "export_encoder_mlir",
    "export_encoder_iree_parameters",
    "import_encoder_dataset_from_hugging_face",
]


def export_encoder_mlir(
    model: Union[T5Encoder, Path, str],
    batch_sizes: list[int],
    mlir_output_path: str,
    dynamic_context_length: bool = True,
):
    """
    Args:
      model: either the torch module or path to GGUF/IRPA.
    """
    if isinstance(model, (Path, str)):
        dataset = Dataset.load(model)
        config = T5Config.from_properties(
            dataset.properties,
        )
        model = T5Encoder(theta=dataset.root_theta, config=config)
    for t in model.theta.flatten().values():
        ExternalTensorTrait(external_name=t.name, external_scope="").set(t.as_torch())

    fxb = FxProgramsBuilder(model)

    for batch_size in batch_sizes:
        _, sample_inputs = model.sample_inputs(batch_size)

        if dynamic_context_length:
            context_length_dim_idx = 1
            assert (
                sample_inputs["input_ids"].shape[context_length_dim_idx]
                % config.context_length_padding_block_size
                == 0
            )
            context_length_block_dim_max = (
                sample_inputs["input_ids"].shape[context_length_dim_idx]
                // config.context_length_padding_block_size
            )
            context_length_block_dim = torch.export.Dim(
                "block", max=context_length_block_dim_max
            )
            context_length_dim = (
                config.context_length_padding_block_size * context_length_block_dim
            )
            dynamic_shapes = {"input_ids": {context_length_dim_idx: context_length_dim}}
        else:
            dynamic_shapes = None

        @fxb.export_program(
            name=f"forward_bs{batch_size}",
            args=tuple(sample_inputs.values()),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        def _(
            model,
            input_ids,
        ):
            return model(input_ids)

    output = export(fxb, import_symbolic_shape_expressions=True)
    output.save_mlir(mlir_output_path)


def export_encoder_iree_parameters(
    model_path_or_dataset: str | Dataset,
    output_path: str,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(model_path_or_dataset, Dataset):
        dataset = copy(model_path_or_dataset)
    else:
        dataset = Dataset.load(model_path_or_dataset)
    if dtype:
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
    dataset.save(output_path)


def import_encoder_dataset_from_hugging_face(
    repo_id_or_model: transformers.T5EncoderModel | str,
    /,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    tokenizer_path_or_repo_id: str | None = None,
    tokenizer_subfolder: str | None = None,
    subfolder: str = "",
) -> Dataset:
    model = repo_id_or_model
    if not isinstance(repo_id_or_model, transformers.T5EncoderModel):
        model = transformers.T5EncoderModel.from_pretrained(
            repo_id_or_model, subfolder=subfolder, torch_dtype="auto"
        )
        from transformers.models.auto.tokenization_auto import get_tokenizer_config

        if tokenizer_path_or_repo_id is None:
            assert not isinstance(repo_id_or_model, transformers.T5EncoderModel)
            tokenizer_path_or_repo_id = repo_id_or_model
        if tokenizer_subfolder is None:
            tokenizer_subfolder = subfolder
        if tokenizer_config is None:
            tokenizer_config = get_tokenizer_config(
                tokenizer_path_or_repo_id, subfolder=tokenizer_subfolder
            )
    else:
        if tokenizer_config is None:
            raise ValueError(
                "When providing a model directly tokenizer_config must also be provided."
            )

    theta = torch_module_to_theta(model)
    config = T5Config.from_hugging_face_config(
        model.config, tokenizer_config=tokenizer_config
    )
    return Dataset(properties=config.to_properties(), root_theta=theta)
