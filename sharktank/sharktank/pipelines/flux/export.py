# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export utilities for Flux text-to-image pipeline."""
#TODO: DO NOT SUBMIT: FIX AND TEST THIS FILE
import functools
from typing import Optional, Union
from pathlib import Path
import torch
from copy import copy

from .flux_pipeline import FluxPipeline
from ...types import Dataset
from ...transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export

__all__ = [
    "export_flux_pipeline_mlir",
    "export_flux_pipeline_iree_parameters",
]

def export_flux_pipeline_mlir(
    model: Union[FluxPipeline, Path, str],
    batch_sizes: list[int],
    mlir_output_path: str,
    dtype: torch.dtype,
):
    """Export Flux pipeline to MLIR format.
    
    Args:
        model: Either the FluxPipeline instance or path to model files
        batch_sizes: List of batch sizes to export for
        mlir_output_path: Output path for MLIR file
    """
    if isinstance(model, (Path, str)):
        model = FluxPipeline(
            t5_path=str(Path(model) / "text_encoder_2/model.gguf"),
            clip_path=str(Path(model) / "text_encoder/model.irpa"),
            transformer_path=str(Path(model) / "transformer/model.irpa"),
            ae_path=str(Path(model) / "vae/model.irpa"),
            dtype=dtype,
        )

    fxb = FxProgramsBuilder(model)

    for batch_size in batch_sizes:
        # Create sample inputs with default dimensions
        t5_prompt_ids = torch.zeros((batch_size, 128), dtype=torch.long)
        clip_prompt_ids = torch.zeros((batch_size, 77), dtype=torch.long)
        latents = model._get_noise(
            1,
            1024,
            1024,
            seed=12345,
        )

        @fxb.export_program(
            name=f"forward_bs{batch_size}",
            args=(t5_prompt_ids, clip_prompt_ids, latents),
            dynamic_shapes={},
            strict=False,
        )
        def _(model, t5_prompt_ids, clip_prompt_ids, latents):
            return model.forward(
                t5_prompt_ids=t5_prompt_ids,
                clip_prompt_ids=clip_prompt_ids,
                latents=latents,
            )

    try:
        output = export(fxb)
    except Exception as e:
        print(f"Error during export: {e}")
        print(f"Model dtype: {model.dtype}")
        print(f"Latents dtype: {latents.dtype}")
        raise
    output.save_mlir(mlir_output_path)

# def export_flux_pipeline_iree_parameters(
#     model_path_or_dataset: str | Dataset,
#     output_path: str,
#     dtype: Optional[torch.dtype] = None,
# ):
#     """Export Flux pipeline parameters to IREE format.
    
#     Args:
#         model_path_or_dataset: Path to model files or Dataset instance
#         output_path: Output path for IREE parameters
#         dtype: Optional dtype to convert parameters to
#     """
#     # TODO: loop over models
#     if isinstance(model_path_or_dataset, Dataset):
#         dataset = copy(model_path_or_dataset)
#     else:
#         dataset = Dataset.load(model_path_or_dataset)
        
#     if dtype:
#         dataset.root_theta = dataset.root_theta.transform(
#             functools.partial(set_float_dtype, dtype=dtype)
#         )
        
#     dataset.save(output_path)