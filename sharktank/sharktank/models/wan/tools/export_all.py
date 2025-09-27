# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import argparse
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import fields
import functools
import os

import torch
import torch.nn.functional as F
from iree.turbine.aot import *
from iree.turbine import aot
import numpy as np
from sharktank.types.theta import torch_module_to_theta, Dataset
from sharktank.transforms.dataset import set_float_dtype

logger = logging.getLogger(__name__)
# --- Helper Functions and Classes ---


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


def save_dataset(path: os.PathLike, model: torch.nn.Module, torch_dtype: torch.dtype):
    """Saves model parameters to a dataset file."""
    theta = torch_module_to_theta(model)
    theta.rename_tensors_to_paths()
    theta.transform(functools.partial(set_float_dtype, dtype=torch_dtype))
    if getattr(model, "config"):
        properties = model.config
    else:
        properties = {}
    ds = Dataset(root_theta=theta, properties={})
    ds.save(path)


# --- Callable Export Function ---


def export_component(
    component: str,
    height: int,
    width: int,
    num_frames: int,
    wan_repo: Optional[str] = None,
    batch_size: int = 1,
    dtype: str = "bf16",
    artifacts_path: os.PathLike = ".",
    return_paths: bool = False,
):
    """
    Exports a single specified model component to MLIR and weights files.

    Args:
        component (str): The name of the component to export.
                         Options: 'clip', 't5', 'vae', 'transformer'.
        height (int): The height of the input frames.
        width (int): The width of the input frames.
        num_frames (int): The number of frames for the transformer/VAE models.
        wan_repo (Optional[str]): The Hugging Face repository ID. Required
                                  for the transformer component.
        batch_size (int): The batch size for all models.
        dtype (str): The data type for the export (e.g., 'bf16').
    """
    if component not in ["clip", "t5", "vae", "transformer"]:
        raise ValueError(
            f"Invalid component '{component}'. Choose from 'clip', 't5', 'vae', 'transformer'."
        )

    logger.info(f"\n🚀 Exporting '{component}' component...")

    dims = f"{width}x{height}"

    # TODO: smarter name construction
    model_name = "wan2_1_1-3b" if "1.3B" in wan_repo else "wan2_1_14b"

    if component == "transformer":
        from sharktank.models.wan.export import (
            export_wan_transformer_from_hugging_face,
        )

        if not wan_repo:
            raise ValueError(
                "The 'wan_repo' argument is required for exporting the 'transformer' component."
            )
        mlir_path = os.path.join(
            artifacts_path, f"{model_name}_transformer_{dims}_{dtype}.mlir"
        )
        weights_path = os.path.join(
            artifacts_path, f"{model_name}_transformer_{dtype}.irpa"
        )
        export_wan_transformer_from_hugging_face(
            repo_id=wan_repo,
            mlir_output_path=mlir_path,
            parameters_output_path=weights_path,
            batch_sizes=[batch_size],
            height=height,
            width=width,
            num_frames=num_frames,
        )
        logger.info(f"✅ Saved Transformer MLIR to: {mlir_path}")
        if return_paths:
            return mlir_path, weights_path
        return

    else:
        raise NotImplementedError()


# --- Main Execution Block ---


def main():
    """Parses command-line arguments and runs the export process."""
    parser = argparse.ArgumentParser(description="Export WANI2V Model Components")
    parser.add_argument(
        "--wan_repo",
        type=str,
        default="wan-AI/Wan2.1-T2V-14B",
        help="Hugging Face model repository (required for transformer).",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--export",
        type=str,
        default="all",
        help="Component(s) to export. Comma-separated: 't5', 'clip', 'vae', 'transformer', or 'all'.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    all_components = ["clip", "t5", "vae", "transformer"]

    if args.export.lower() == "all":
        components_to_export = all_components
    else:
        components_to_export = [
            c.strip() for c in args.export.split(",") if c.strip() in all_components
        ]
        if not components_to_export:
            logger.error(
                f"No valid components specified. Please choose from: {', '.join(all_components)}"
            )
            return

    for component in components_to_export:
        try:
            export_component(
                component=component,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                wan_repo=args.wan_repo,
                batch_size=args.batch_size,
            )
        except Exception as e:
            logger.error(f"❌ Failed to export '{component}': {e}")


if __name__ == "__main__":
    main()
