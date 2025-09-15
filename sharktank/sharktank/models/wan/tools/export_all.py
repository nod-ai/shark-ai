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


def transform_normalize(x):
    """Normalizes a tensor with pre-defined mean and std."""
    mean = torch.as_tensor(
        [0.48145466, 0.4578275, 0.40821073], dtype=torch.bfloat16
    ).view(-1, 1, 1)
    std = torch.as_tensor(
        [0.26862954, 0.26130258, 0.27577711], dtype=torch.bfloat16
    ).view(-1, 1, 1)
    return x.sub_(mean).div_(std)


class ExportSafeClipModel(torch.nn.Module):
    """A wrapper for the CLIP model to handle pre-processing during export."""

    def __init__(self, mod):
        super().__init__()
        self.model = mod
        self.size = (self.model.image_size, self.model.image_size)

    def forward(self, video_1, video_2):
        videos = [
            F.interpolate(
                video_1.transpose(0, 1).type(torch.float16),
                size=self.size,
                mode="bicubic",
                align_corners=False,
            ),
            F.interpolate(
                video_2.transpose(0, 1).type(torch.float16),
                size=self.size,
                mode="bicubic",
                align_corners=False,
            ),
        ]
        videos = torch.cat(videos)
        videos = transform_normalize(videos.mul_(0.5).add_(0.5)).to(torch.bfloat16)
        out = self.model.visual(videos, use_31_block=True)
        return out


class WanVaeWrapped(torch.nn.Module):
    """A wrapper for the VAE model to expose 'encode' and 'decode' methods for fp16 inputs."""

    def __init__(self, mod, dtype=torch.bfloat16):
        super().__init__()
        self.model = mod
        self.inner_dtype = dtype

    def encode(self, x):
        x = x.to(self.inner_dtype)
        return self.model.encode(x).latent_dist.mode()

    def decode(self, z):
        z = z.to(self.inner_dtype)
        return self.model.decode(z, return_dict=False)


def get_clip_visual_model_and_inputs(height: int, width: int):
    """Initializes the CLIP model and generates sample inputs."""
    from sharktank.models.wan.clip_ref import clip_xlm_roberta_vit_h_14

    inner = (
        clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=False,
            return_tokenizer=False,
            dtype=torch.bfloat16,
        )
        .eval()
        .requires_grad_(False)
    )
    mod = ExportSafeClipModel(inner).eval().requires_grad_(False)
    mod.model.log_scale = torch.nn.Parameter(
        mod.model.log_scale.to(torch.float32)
    ).requires_grad_(False)

    inputs = {
        "forward": {
            "video_1": torch.rand(3, 1, height, width, dtype=torch.float16),
            "video_2": torch.rand(3, 1, height, width, dtype=torch.float16),
        }
    }
    return mod, inputs


def get_t5_text_model_and_inputs(batch_size=1, artifacts_dir="."):
    from sharktank.models.t5.export import import_encoder_dataset_from_hugging_face
    from sharktank.models.t5 import T5Config, T5Encoder

    model_path = "google/umt5-xxl"
    dtype_str = "bf16"
    output_path = artifacts_dir
    t5_path = Path(model_path)
    t5_tokenizer_path = Path(model_path)
    t5_output_path = Path(output_path) / f"wan2_1_t5_{dtype_str}.irpa"
    t5_dataset = import_encoder_dataset_from_hugging_face(
        str(t5_path), tokenizer_path_or_repo_id=str(t5_tokenizer_path)
    )
    t5_dataset.properties = filter_properties_for_config(
        t5_dataset.properties, T5Config
    )
    t5_dataset.save(str(t5_output_path))

    class HFEmbedder(torch.nn.Module):
        def __init__(self, dataset, max_length: int):
            super().__init__()
            self.max_length = max_length
            self.output_key = "last_hidden_state"

            t5_config = T5Config.from_properties(dataset.properties)
            self.hf_module = T5Encoder(theta=dataset.root_theta, config=t5_config)

            self.hf_module = self.hf_module.eval().requires_grad_(False)

        def forward(self, input_ids) -> torch.Tensor:

            outputs = self.hf_module(
                input_ids=input_ids,
                attention_mask=None,
                output_hidden_states=False,
            )
            return outputs[self.output_key]

    t5_mod = HFEmbedder(
        t5_dataset,
        512,
    )
    t5_sample_inputs = {
        "forward": {
            "input_ids": torch.ones([batch_size, 512], dtype=torch.int64),
        }
    }
    # start = time.time()
    # t5_output = t5_mod.forward(t5_sample_inputs["forward"]["input_ids"])
    # end = time.time()
    # print("umt5xxl baseline performance: ", end - start, " seconds")
    # np.save("umt5xxl_input.npy", np.asarray(t5_sample_inputs["forward"]["input_ids"]))

    # np.save("umt5xxl_output.npy", np.asarray(t5_output.to(torch.float16)))
    return t5_mod, t5_sample_inputs


def get_vae_model_and_inputs(height: int, width: int):
    """Initializes the VAE model and generates sample inputs."""
    from diffusers import AutoencoderKLWan

    # cfg = dict(
    #     dim=96,
    #     z_dim=16,
    #     dim_mult=[1, 2, 4, 4],
    #     num_res_blocks=2,
    #     attn_scales=[],
    #     temperal_downsample=[False, True, True],
    #     dropout=0.0,
    # )
    model = (
        AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae"
        )
        .bfloat16()
        .requires_grad_(False)
        .eval()
    )
    wrapped_model = WanVaeWrapped(model)
    inputs = {
        "encode": {"x": torch.rand(1, 3, 1, height, width, dtype=torch.float16)},
        "decode": {
            "z": torch.rand(1, 16, 21, height // 8, width // 8, dtype=torch.float16)
        },
    }
    return wrapped_model, inputs


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

    # Map component names to their setup functions
    GET_MODEL_MAP = {
        "clip": (get_clip_visual_model_and_inputs, {"height": height, "width": width}),
        "t5": (
            get_t5_text_model_and_inputs,
            {"batch_size": batch_size, "artifacts_dir": artifacts_path},
        ),
        "vae": (get_vae_model_and_inputs, {"height": height, "width": width}),
    }

    # Get the appropriate model and inputs
    model_func, kwargs = GET_MODEL_MAP[component]
    model, inputs = model_func(**kwargs)

    # Define artifact names
    mlir_path = os.path.join(
        artifacts_path, f"{model_name}_{component}_{dims}_{dtype}.mlir"
    )
    weights_path = os.path.join(
        artifacts_path, f"{model_name}_{component}_{dtype}.irpa"
    )

    # Export the model
    export_model_mlir(
        model,
        mlir_path,
        inputs,
        decomp_attn=(component == "clip"),
        weights_filename=weights_path,
    )
    if return_paths:
        return mlir_path, weights_path


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
