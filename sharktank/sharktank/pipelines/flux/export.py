# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export utilities for Flux text-to-image pipeline."""
import functools
from typing import Optional, Union
from pathlib import Path
import torch
from copy import copy
import logging

from .flux_pipeline import FluxPipeline
from ...types import Dataset, dtype_to_serialized_short_name
from ...transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export
from ...models.t5.export import export_encoder_iree_parameters as export_t5_parameters
from ...models.clip.export import export_clip_text_model_iree_parameters
from ...models.flux.export import export_flux_transformer_iree_parameters
from ...models.vae.model import VaeDecoderModel
from ...models.clip import ClipTextModel, ClipTextConfig
from transformers import CLIPTokenizer, T5Tokenizer, CLIPTextModel as HfCLIPTextModel
from ...models.flux.flux import FluxModelV1, FluxParams

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
        model_parameter_path = Path(model) / f"exported_parameters_{dtype_to_serialized_short_name(dtype)}"
        model = FluxPipeline(
            t5_path=str(model_parameter_path / "t5.irpa"),
            clip_path=str(model_parameter_path / "clip.irpa"),
            transformer_path=str(model_parameter_path / "transformer.irpa"),
            ae_path=str(model_parameter_path / "vae.irpa"),
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

def is_already_exported(output_path: Path) -> bool:
    return output_path.exists()

def export_flux_pipeline_iree_parameters(
    model_path_or_dataset: str | Dataset,
    output_path: str,
    dtype: Optional[torch.dtype] = None,
):
    """Export Flux pipeline parameters to IREE format.
    
    Args:
        model_path_or_dataset: Path to model files or Dataset instance
        output_path: Output path for IREE parameters
        dtype: Optional dtype to convert parameters to
    """
    # Ensure output_path is a Path object
    output_path = Path(output_path) / f"exported_parameters_{dtype_to_serialized_short_name(dtype)}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Export T5 parameters
    t5_path = Path(model_path_or_dataset) / "text_encoder_2/model.gguf"
    t5_output_path = output_path / "t5.irpa"
    print("hi")
    if not is_already_exported(t5_output_path):
        print("hello")
        export_t5_parameters(t5_path, str(t5_output_path), dtype)
        logging.info(f"Exported T5 parameters to {t5_output_path}")
    else:
        logging.info(f"Skipped T5 parameter export, already exists at {t5_output_path}")

    # Export CLIP parameters
    clip_path = Path(model_path_or_dataset) / "text_encoder/model.irpa"
    clip_output_path = output_path / "clip.irpa"
    if not is_already_exported(clip_output_path):
        clip_dataset = Dataset.load(clip_path)
        # TODO: Refactor CLIP to not make the config rely on HuggingFace
        hf_clip_model = HfCLIPTextModel.from_pretrained("/data/flux/FLUX.1-dev/text_encoder/")
        clip_config = ClipTextConfig.from_hugging_face_clip_text_model_config(hf_clip_model.config)
        clip_model = ClipTextModel(theta=clip_dataset.root_theta, config=clip_config)
        export_clip_text_model_iree_parameters(clip_model, str(clip_output_path))
        logging.info(f"Exported CLIP parameters to {clip_output_path}")
    else:
        logging.info(f"Skipped CLIP parameter export, already exists at {clip_output_path}")

    # Export FluxTransformer parameters
    transformer_path = Path(model_path_or_dataset) / "transformer/model.irpa"
    transformer_output_path = output_path / "transformer.irpa"
    if not is_already_exported(transformer_output_path):
        transformer_dataset = Dataset.load(transformer_path)
        transformer_model = FluxModelV1(theta=transformer_dataset.root_theta, params=FluxParams.from_hugging_face_properties(transformer_dataset.properties))
        export_flux_transformer_iree_parameters(transformer_model, str(transformer_output_path), dtype=dtype)
        logging.info(f"Exported FluxTransformer parameters to {transformer_output_path}")
    else:
        logging.info(f"Skipped FluxTransformer parameter export, already exists at {transformer_output_path}")

    # Export VAE parameters
    vae_path = Path(model_path_or_dataset) / "vae/model.irpa"
    vae_output_path = output_path / "vae.irpa"
    if not is_already_exported(vae_output_path):
        vae_dataset = Dataset.load(vae_path)
        vae_dataset.root_theta = vae_dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=dtype)
        )
        vae_dataset.save(str(vae_output_path))
        logging.info(f"Exported VAE parameters to {vae_output_path}")
    else:
        logging.info(f"Skipped VAE parameter export, already exists at {vae_output_path}")

    logging.info(f"Completed Flux pipeline parameter export to {output_path}")