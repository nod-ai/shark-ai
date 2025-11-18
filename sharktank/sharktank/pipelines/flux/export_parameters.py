# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export utilities for Flux text-to-image pipeline."""
import argparse
from copy import copy
import functools
import logging
from typing import Any, Dict, Optional, Set, Union
from pathlib import Path
from dataclasses import fields

import torch
from transformers import CLIPTextModel as HfCLIPTextModel

from amdsharktank.tools.import_hf_dataset import import_hf_dataset
from amdsharktank.models.t5 import T5Config
from amdsharktank.models.clip import ClipTextConfig
from amdsharktank.models.flux.flux import FluxParams
from amdsharktank.types import Dataset, dtype_to_serialized_short_name
from amdsharktank.transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export
from amdsharktank.models import t5
from amdsharktank.models.clip.export import (
    export_clip_text_model_dataset_from_hugging_face,
)
from amdsharktank.models.flux.export import export_flux_transformer_iree_parameters
from amdsharktank.models.vae.model import VaeDecoderModel
from amdsharktank.models.clip import ClipTextModel, ClipTextConfig
from amdsharktank.models.flux.flux import FluxModelV1, FluxParams

from .flux_pipeline import FluxPipeline

__all__ = [
    "export_flux_pipeline_iree_parameters",
]

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def is_already_exported(output_path: Path) -> bool:
    return output_path.exists()


def find_safetensors_files(path: Path) -> list[Path]:
    """Find all .safetensors files in a directory, excluding index files."""
    safetensors_files = list(path.glob("*.safetensors"))
    safetensors_files.sort()
    return safetensors_files


def filter_properties_for_config(
    properties: Dict[str, Any], config_class: Any
) -> Dict[str, Any]:
    """Filter properties to only include fields valid for the given config class.

    Args:
        properties: Properties dictionary
        config_class: The dataclass to filter properties for

    Returns:
        Filtered properties dictionary with only valid fields for the config class
    """
    # Start with hparams if available
    if "hparams" in properties:
        props = properties["hparams"]
    else:
        props = properties

    # Get set of valid field names for the config class
    valid_fields = {f.name for f in fields(config_class)}

    # Filter to only include valid fields
    filtered_props = {k: v for k, v in props.items() if k in valid_fields}

    return filtered_props


def export_flux_pipeline_iree_parameters(
    model_path: str,
    output_path: str,
    model_name: Optional[str] = None,
    dtype_str: Optional[str] = None,
):
    """Export Flux pipeline parameters to IREE format.

    Args:
        model_path: Path to model files
        output_path: Output path for IREE parameters
        dtype: Optional dtype to convert parameters to
    """
    # Ensure output_path is a Path object
    if not model_name:
        model_name = "flux_dev"
    if not dtype_str:
        dtype_str = "bf16"
    dtype = torch_dtypes[dtype_str]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export FluxTransformer parameters
    transformer_path = Path(model_path) / "transformer/"
    transformer_output_path = output_path / f"{model_name}_sampler_{dtype_str}.irpa"
    if not is_already_exported(transformer_output_path):
        config_json_path = transformer_path / "config.json"
        st_str = model_name.replace("flux_", "flux1-")
        param_paths = [Path(model_path) / f"{st_str}.safetensors"]
        transformer_dataset = import_hf_dataset(
            config_json_path, param_paths, target_dtype=dtype
        )
        transformer_dataset.save(str(transformer_output_path))
        logging.info(
            f"Exported FluxTransformer parameters to {transformer_output_path}"
        )
    else:
        logging.info(
            f"Skipped FluxTransformer parameter export, already exists at {transformer_output_path}"
        )

    # Export T5 parameters
    t5_path = Path(model_path) / "text_encoder_2/"
    t5_tokenizer_path = Path(model_path) / "tokenizer_2/"
    t5_output_path = output_path / f"{model_name.split('_')[0]}_t5xxl_{dtype_str}.irpa"
    if not is_already_exported(t5_output_path):
        config_json_path = t5_path / "config.json"
        param_paths = find_safetensors_files(t5_path)
        t5_dataset = t5.export.import_encoder_dataset_from_hugging_face(
            str(t5_path), tokenizer_path_or_repo_id=str(t5_tokenizer_path)
        )
        t5_dataset.properties = filter_properties_for_config(
            t5_dataset.properties, T5Config
        )
        t5_dataset.save(str(t5_output_path))
        logging.info(f"Exported T5 parameters to {t5_output_path}")
    else:
        logging.info(f"Skipped T5 parameter export, already exists at {t5_output_path}")

    # Export CLIP parameters
    clip_path = Path(model_path) / "text_encoder/"
    clip_output_path = output_path / f"{model_name.split('_')[0]}_clip_{dtype_str}.irpa"
    if not is_already_exported(clip_output_path):
        config_json_path = clip_path / "config.json"
        param_paths = find_safetensors_files(clip_path)
        clip_dataset = import_hf_dataset(
            config_json_path, param_paths, target_dtype=dtype
        )
        clip_dataset.properties = filter_properties_for_config(
            clip_dataset.properties, ClipTextConfig
        )
        clip_dataset.save(str(clip_output_path))
        logging.info(f"Exported CLIP parameters to {clip_output_path}")
    else:
        logging.info(
            f"Skipped CLIP parameter export, already exists at {clip_output_path}"
        )

    # Export VAE parameters
    vae_path = Path(model_path) / "vae/"
    vae_output_path = output_path / f"{model_name.split('_')[0]}_vae_{dtype_str}.irpa"
    if not is_already_exported(vae_output_path):
        config_json_path = vae_path / "config.json"
        param_paths = find_safetensors_files(vae_path)
        import_hf_dataset(
            config_json_path, param_paths, vae_output_path, target_dtype=dtype
        )
        logging.info(f"Exported VAE parameters to {vae_output_path}")
    else:
        logging.info(
            f"Skipped VAE parameter export, already exists at {vae_output_path}"
        )

    logging.info(f"Completed Flux pipeline parameter export to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--input-path", type=str, default="/data/flux/FLUX.1-dev/")
    parser.add_argument("--output-path", type=str, default="/data/flux/FLUX.1-dev/")
    parser.add_argument(
        "--model",
        default="flux_schnell",
        choices=["flux_dev", "flux_schnell", "flux_pro"],
    )
    args = parser.parse_args()
    export_flux_pipeline_iree_parameters(
        args.input_path,
        args.output_path,
        model_name=args.model,
        dtype_str=args.dtype,
    )


if __name__ == "__main__":
    main()
