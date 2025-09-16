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


# Set a seed for reproducibility
torch.random.manual_seed(0)

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


def save_dataset(path, model):
    """Saves model parameters to a dataset file."""
    theta = torch_module_to_theta(model)
    theta.rename_tensors_to_paths()
    theta.transform(functools.partial(set_float_dtype, dtype=torch.bfloat16))
    ds = Dataset(root_theta=theta, properties={})
    ds.save(path)


def export_model_mlir(
    model,
    output_path: str,
    function_inputs_map: Dict,
    decomp_attn: bool = False,
    weights_filename: str = "parameters.irpa",
):
    """Exports a PyTorch model to MLIR using Turbine AOT."""
    decomp_list = [
        torch.ops.aten.logspace,
        torch.ops.aten.upsample_bicubic2d.vec,
        torch.ops.aten._upsample_nearest_exact2d.vec,
        torch.ops.aten.as_strided,
        torch.ops.aten.as_strided_copy.default,
    ]
    if decomp_attn:
        decomp_list.extend(
            [
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops.aten.scaled_dot_product_attention,
            ]
        )

    with aot.decompositions.extend_aot_decompositions(
        from_current=True, add_ops=decomp_list
    ):
        save_dataset(weights_filename, model)
        aot.externalize_module_parameters(model)
        fxb = aot.FxProgramsBuilder(model)

        for function, input_kwargs in function_inputs_map.items():

            @fxb.export_program(
                name=f"{function or 'forward'}",
                args=(),
                kwargs=input_kwargs,
                strict=False,
            )
            def _(mdl, **kwargs):
                return getattr(mdl, function, mdl.forward)(**kwargs)

        output = aot.export(fxb)
        output.save_mlir(output_path)
        print(f"✅ Saved MLIR to: {output_path}")


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

    print(f"\n🚀 Exporting '{component}' component...")

    dims = f"{width}x{height}"
    model_name = "wan2_1"

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
        print(f"✅ Saved Transformer MLIR to: {mlir_path}")
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
            print(
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
            print(f"❌ Failed to export '{component}': {e}")


if __name__ == "__main__":
    main()
