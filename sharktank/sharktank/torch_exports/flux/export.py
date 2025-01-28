# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
from dataclasses import dataclass
import math
import torch
from typing import Callable

from einops import rearrange

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)

from transformers import CLIPTextModel
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.models.t5 import T5Encoder, T5Config
from sharktank.models.flux.flux import FluxModelV1, FluxParams
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.types.theta import Theta, Dataset, torch_module_to_theta



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


class FluxDenoiseStepModel(torch.nn.Module):
    def __init__(
        self,
        theta,
        params,
        batch_size=1,
        max_length=512,
        height=1024,
        width=1024,
    ):
        super().__init__()
        self.mmdit = FluxModelV1(theta=theta, params=params)
        self.batch_size = batch_size
        img_ids = torch.zeros(height // 16, width // 16, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 16)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 16)[None, :]
        self.img_ids = img_ids.reshape(1, height * width // 256, 3)
        self.txt_ids = torch.zeros(1, max_length, 3)

    def forward(self, img, txt, vec, step, timesteps, guidance_scale):
        guidance_vec = guidance_scale.repeat(self.batch_size)
        t_curr = torch.index_select(timesteps, 0, step)
        t_prev = torch.index_select(timesteps, 0, step + 1)
        t_vec = t_curr.repeat(self.batch_size)

        pred = self.mmdit(
            img=img,
            img_ids=self.img_ids,
            txt=txt,
            txt_ids=self.txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        # TODO: Use guidance scale
        # pred_uncond, pred = torch.chunk(pred, 2, dim=0)
        # pred = pred_uncond + guidance_scale * (pred - pred_uncond)
        img = img + (t_prev - t_curr) * pred
        return img


@torch.no_grad()
def get_flux_transformer_model(
    hf_model_path,
    img_height=1024,
    img_width=1024,
    compression_factor=8,
    max_len=512,
    torch_dtype=torch.float32,
    bs=1,
):
    # DNS: refactor file to select datatype
    transformer_dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/transformer.irpa")
    model = FluxDenoiseStepModel(theta=transformer_dataset.root_theta, params=FluxParams.from_hugging_face_properties(transformer_dataset.properties))
    sample_args, sample_kwargs = model.mmdit.sample_inputs()
    sample_inputs = (
        sample_kwargs["img"],
        sample_kwargs["txt"],
        sample_kwargs["y"],
        torch.full((bs,), 1, dtype=torch.int64),
        torch.full((100,), 1, dtype=torch_dtype), # TODO: non-dev timestep sizes
        sample_kwargs["guidance"],
    )
    return model, sample_inputs


def get_flux_model_and_inputs(
    hf_model_name, precision, batch_size, max_length, height, width
):
    dtype = torch_dtypes[precision]
    return get_flux_transformer_model(
        hf_model_name, height, width, 8, max_length, dtype, batch_size
    )

# Copied from https://github.com/black-forest-labs/flux
class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, **hf_kwargs
            )
            # DNS: Refactor to not rely on huggingface
            config = ClipTextConfig.from_hugging_face_clip_text_model_config(self.hf_module.config)
            config.dtype = torch.float32
            dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/clip.irpa")
            self.hf_module = ClipTextModel(theta=dataset.root_theta, config=config)
        else:
            t5_dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/t5.irpa")
            t5_config = T5Config.from_gguf_properties(
                t5_dataset.properties,
                feed_forward_proj="gated-gelu",
            )
            self.hf_module = T5Encoder(theta=t5_dataset.root_theta, config=t5_config)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, input_ids) -> Tensor:
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

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
            te = HFEmbedder(
                "t5xxl",
                max_length=512,
                torch_dtype=torch.float32,
            )
            clip_ids_shape = (
                batch_size,
                512, #DNS
            )
            input_args = [
                torch.ones(clip_ids_shape, dtype=torch.int64),
            ]
            return te, input_args


class FluxAEWrapper(torch.nn.Module):
    def __init__(self, height=1024, width=1024, precision="fp32"):
        super().__init__()
        dtype = torch_dtypes[precision]
        dataset = Dataset.load("/data/flux/flux/FLUX.1-dev/exported_parameters_f32/vae.irpa")
        self.ae = VaeDecoderModel.from_dataset(dataset)
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
        #d_in = d_in / self.ae.config.scaling_factor + self.ae.config.shift_factor
        #return self.ae.decode(d_in, return_dict=False)[0].clamp(-1, 1)
        return self.ae.forward(d_in)


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


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps


class FluxScheduler(torch.nn.Module):
    def __init__(self, max_length, torch_dtype, is_schnell=False):
        super().__init__()
        self.is_schnell = is_schnell
        self.max_length = max_length
        timesteps = [torch.empty((100), dtype=torch_dtype, requires_grad=False)] * 100
        for i in range(1, 100):
            schedule = get_schedule(i, max_length, shift=not self.is_schnell)
            timesteps[i] = torch.nn.functional.pad(schedule, (0, 99 - i), "constant", 0)
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()

    def prepare(self, num_steps):
        timesteps = self.timesteps[num_steps]
        return timesteps

def get_scheduler_model_and_inputs(hf_model_name, max_length, precision):
    is_schnell = "schnell" in hf_model_name
    mod = FluxScheduler(
        max_length=max_length,
        torch_dtype=torch_dtypes[precision],
        is_schnell=is_schnell,
    )
    sample_inputs = (torch.empty(1, dtype=torch.int64),)
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
            model, sample_inputs = get_flux_model_and_inputs(
                hf_model_name, precision, batch_size, max_length, height, width
            )
            print(sample_inputs)

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

        elif component == "clip":
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
        elif component == "t5xxl":
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

            class CompiledFluxTextEncoder2(CompiledModule):
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
        case "t5xxl":
            return create_safe_name(
                args.model, f"t5xxl_bs{args.batch_size}_256_{args.precision}"
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
