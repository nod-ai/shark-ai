# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
from dataclasses import dataclass
import math

from einops import rearrange

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
import torch

from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from te import HFEmbedder
from transformers import CLIPTextModel
from ae import AutoEncoder, AutoEncoderParams
from scheduler import FluxScheduler
from mmdit import get_flux_transformer_model


@dataclass
class ModelSpec:
    ae_params: AutoEncoderParams
    ae_path: str | None


fluxconfigs = {
    "flux-dev": ModelSpec(
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            height=1024,
            width=1024,
        ),
    ),
    "flux-schnell": ModelSpec(
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
            height=1024,
            width=1024,
        ),
    ),
}

model_repo_map = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}
model_file_map = {
    "flux-dev": "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors",
    "flux-schnell": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/flux1-schnell.safetensors",
}

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


def get_flux_model_and_inputs(
    hf_model_name, precision, batch_size, max_length, height, width
):
    dtype = torch_dtypes[precision]
    return get_flux_transformer_model(
        hf_model_name, height, width, 8, max_length, dtype, batch_size
    )


def get_te_model_and_inputs(
    hf_model_name, component, precision, batch_size, max_length
):
    match component:
        case "clip":
            te = HFEmbedder(
                "openai/clip-vit-large-patch14",
                max_length=77,
                torch_dtype=torch.float32,
            )
            clip_ids_shape = (
                batch_size,
                77,
            )
            input_args = [
                torch.ones(clip_ids_shape, dtype=torch.int64),
            ]
            return te, input_args
        case "t5xxl":
            return None, None


class FluxAEWrapper(torch.nn.Module):
    def __init__(self, height=1024, width=1024, precision="fp32"):
        super().__init__()
        dtype = torch_dtypes[precision]
        self.ae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtypes=dtype
        )
        self.height = height
        self.width = width

    def forward(self, z):
        d_in = rearrange(
            z,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(self.height / 16),
            w=math.ceil(self.width / 16),
            ph=2,
            pw=2,
        )
        d_in = d_in / self.ae.config.scaling_factor + self.ae.config.shift_factor
        return self.ae.decode(d_in, return_dict=False)[0].clamp(-1, 1)


def get_ae_model_and_inputs(hf_model_name, precision, batch_size, height, width):
    dtype = torch_dtypes[precision]
    aeparams = fluxconfigs[hf_model_name].ae_params
    aeparams.height = height
    aeparams.width = width
    ae = FluxAEWrapper(height, width, precision).to(dtype)
    latents_shape = (
        batch_size,
        int(height * width / 256),
        64,
    )
    img_shape = (
        1,
        aeparams.in_channels,
        int(height),
        int(width),
    )
    encode_inputs = [
        torch.empty(img_shape, dtype=dtype),
    ]
    decode_inputs = [
        torch.empty(latents_shape, dtype=dtype),
    ]
    return ae, encode_inputs, decode_inputs


def get_scheduler_model_and_inputs(hf_model_name, max_length, precision):
    is_schnell = "schnell" in hf_model_name
    mod = FluxScheduler(
        max_length=max_length,
        torch_dtype=torch_dtypes[precision],
        is_schnell=is_schnell,
    )
    sample_inputs = (torch.empty(1, dtype=torch.int64),)
    # tdim = torch.export.Dim("timesteps")
    # dynamic_inputs = {"timesteps": {0: tdim}}
    return mod, sample_inputs


@torch.no_grad()
def export_flux_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=512,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    decomp_attn=False,
):
    dtype = torch_dtypes[precision]
    decomp_list = []
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if component == "mmdit":
            model, sample_inputs, _ = get_flux_model_and_inputs(
                hf_model_name, precision, batch_size, max_length, height, width
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxTransformer(CompiledModule):
                run_forward = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledFluxTransformer(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)

        elif component in ["clip", "t5xxl"]:
            model, sample_inputs = get_te_model_and_inputs(
                hf_model_name, component, precision, batch_size, max_length
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxTextEncoder(CompiledModule):
                encode_prompts = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledFluxTextEncoder(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
        elif component == "vae":
            model, encode_inputs, decode_inputs = get_ae_model_and_inputs(
                hf_model_name, precision, batch_size, height, width
            )

            fxb = FxProgramsBuilder(model)

            # @fxb.export_program(
            #     args=(encode_inputs,),
            # )
            # def _encode(
            #     module,
            #     inputs,
            # ):
            #     return module.encode(*inputs)

            @fxb.export_program(
                args=(decode_inputs,),
            )
            def _decode(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledFluxAutoEncoder(CompiledModule):
                # encode = _encode
                decode = _decode

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledFluxAutoEncoder(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)

        elif component == "scheduler":
            model, sample_inputs = get_scheduler_model_and_inputs(
                hf_model_name, max_length, precision
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_inputs,),
            )
            def _prepare(
                module,
                inputs,
            ):
                return module.prepare(*inputs)

            class CompiledFlowScheduler(CompiledModule):
                run_prep = _prepare

            inst = CompiledFlowScheduler(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)

    module_str = str(module)
    return module_str


def get_filename(args):
    match args.component:
        case "mmdit":
            return create_safe_name(
                args.model,
                f"mmdit_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "clip":
            return create_safe_name(
                args.model, f"clip_bs{args.batch_size}_77_{args.precision}"
            )
        case "scheduler":
            return create_safe_name(
                args.model,
                f"scheduler_bs{args.batch_size}_{args.max_length}_{args.precision}",
            )
        case "vae":
            return create_safe_name(
                args.model,
                f"vae_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}",
            )


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell", "flux-pro"],
    )
    p.add_argument(
        "--component",
        default="mmdit",
        choices=["mmdit", "clip", "t5xxl", "scheduler", "vae"],
    )
    p.add_argument("--batch_size", default=1)
    p.add_argument("--height", default=1024)
    p.add_argument("--width", default=1024)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--max_length", default=512)
    p.add_argument("--external_weights", default="irpa")
    p.add_argument("--external_weights_file", default=None)
    p.add_argument("--decomp_attn", action="store_true")
    args = p.parse_args()

    if args.external_weights and not args.external_weights_file:
        args.external_weights_file = (
            create_safe_name(
                args.model,
                args.component + "_" + args.precision,
            )
            + "."
            + args.external_weights
        )
    safe_name = get_filename(args)
    mod_str = export_flux_model(
        args.model,
        args.component,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        "mlir",
        args.external_weights,
        args.external_weights_file,
        args.decomp_attn,
    )

    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
