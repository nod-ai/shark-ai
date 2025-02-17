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
from typing import Optional, Union
from pathlib import Path

import torch
from transformers import CLIPTextModel as HfCLIPTextModel

from sharktank.types import Dataset, dtype_to_serialized_short_name
from sharktank.transforms.dataset import set_float_dtype
from iree.turbine.aot import FxProgramsBuilder, export
from sharktank.models.t5.export import (
    export_encoder_iree_parameters as export_t5_parameters,
)
from sharktank.models.clip.export import (
    export_clip_text_model_dataset_from_hugging_face,
)
from sharktank.models.flux.export import export_flux_transformer_iree_parameters
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.models.flux.flux import FluxModelV1, FluxParams

from .flux_pipeline import FluxPipeline

__all__ = [
    "export_flux_pipeline_iree_parameters",
]


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
    output_path = (
        Path(output_path)
        / f"exported_parameters_{dtype_to_serialized_short_name(dtype)}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # Export T5 parameters
    t5_path = Path(model_path_or_dataset) / "text_encoder_2/model.irpa"
    t5_output_path = output_path / "t5.irpa"
    if not is_already_exported(t5_output_path):
        export_t5_parameters(t5_path, str(t5_output_path), dtype)
        logging.info(f"Exported T5 parameters to {t5_output_path}")
    else:
        logging.info(f"Skipped T5 parameter export, already exists at {t5_output_path}")

    # Export CLIP parameters
    clip_path = Path(model_path_or_dataset) / "text_encoder/"
    clip_output_path = output_path / "clip.irpa"
    if not is_already_exported(clip_output_path):
        clip_model = HfCLIPTextModel.from_pretrained(clip_path, torch_dtype=dtype)
        export_clip_text_model_dataset_from_hugging_face(
            clip_model, str(clip_output_path)
        )
        logging.info(f"Exported CLIP parameters to {clip_output_path}")
    else:
        logging.info(
            f"Skipped CLIP parameter export, already exists at {clip_output_path}"
        )

    # Export FluxTransformer parameters
    transformer_path = Path(model_path_or_dataset) / "transformer/model.irpa"
    transformer_output_path = output_path / "transformer.irpa"
    if not is_already_exported(transformer_output_path):
        transformer_dataset = Dataset.load(transformer_path)
        transformer_model = FluxModelV1(
            theta=transformer_dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(
                transformer_dataset.properties
            ),
        )
        export_flux_transformer_iree_parameters(
            transformer_model, str(transformer_output_path), dtype=dtype
        )
        logging.info(
            f"Exported FluxTransformer parameters to {transformer_output_path}"
        )
    else:
        logging.info(
            f"Skipped FluxTransformer parameter export, already exists at {transformer_output_path}"
        )

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
        logging.info(
            f"Skipped VAE parameter export, already exists at {vae_output_path}"
        )

    logging.info(f"Completed Flux pipeline parameter export to {output_path}")


torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--input-path", type=str, default="/data/flux/FLUX.1-dev/")
    parser.add_argument("--output-path", type=str, default="/data/flux/FLUX.1-dev/")
    args = parser.parse_args()
    export_flux_pipeline_iree_parameters(
        args.input_path,
        args.output_path,
        dtype=torch_dtypes[args.dtype],
    )


if __name__ == "__main__":
    main()
