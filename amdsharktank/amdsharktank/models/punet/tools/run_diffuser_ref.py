# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from amdsharktank import ops
from amdsharktank.utils.patching import SaveModuleResultTensorsPatch
from amdsharktank.utils import cli
from .sample_data import get_random_inputs, load_inputs, save_outputs


class ClassifierFreeGuidanceUnetModel(torch.nn.Module):
    def __init__(self, cond_model):
        super().__init__()
        self.cond_model = cond_model

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
        guidance_scale: torch.Tensor,
    ):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        latent_model_input = ops.cat([sample] * 2)
        noise_pred, *_ = self.cond_model.forward(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


def main():

    parser = cli.create_parser()
    parser.add_argument("--device", default="cuda:0", help="Torch device to run on")
    parser.add_argument("--dtype", default="float16", help="DType to run in")
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Safetensors file of inputs (or random if not given)",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        help="Safetensors file of outputs",
    )
    parser.add_argument(
        "--save-intermediates-path",
        type=Path,
        help="Path of safetensors file in which to save all module outputs",
    )
    args = cli.parse(parser)

    device = args.device
    dtype = getattr(torch, args.dtype)

    cond_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        low_cpu_mem_usage=False,
        variant="fp16",
        torch_dtype=dtype,
    )
    print("Created model")
    cond_unet = cond_unet.to(device)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)
    print("Moved to GPU")

    # Run a step for debugging.
    if args.inputs:
        inputs = load_inputs(args.inputs, dtype=dtype, device=device)
    else:
        inputs = get_random_inputs(dtype=dtype, device=device)

    # Save intermediates.
    intermediates_saver = None
    if args.save_intermediates_path:
        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(mdl.cond_model)

    print("Calling forward")
    results = mdl.forward(**inputs)
    print("1-step resutls:", results)
    if args.outputs:
        print(f"Saving outputs to {args.outputs}")
        save_outputs(args.outputs, results)

    if intermediates_saver:
        print(f"Saving intermediate tensors to: {args.save_intermediates_path}")
        intermediates_saver.save_file(args.save_intermediates_path)


if __name__ == "__main__":
    main()
