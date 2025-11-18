# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from iree.turbine.aot import *

from diffusers.models import AutoencoderKL

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        custom_vae="",
    ):
        super().__init__()
        self.vae = None
        if custom_vae in ["", None]:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
        elif isinstance(custom_vae, str) and "safetensors" in custom_vae:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfoler="vae",
            )
            with safe_open(custom_vae, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                self.vae.load_state_dict(state_dict)
        elif not isinstance(custom_vae, dict):
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            fp16_weights = hf_hub_download(
                repo_id=custom_vae,
                filename="vae/vae.safetensors",
            )
            with safe_open(fp16_weights, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                self.vae.load_state_dict(state_dict)
        else:
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            self.vae.load_state_dict(custom_vae)

    def decode(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        x = self.vae.decode(latents, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return self.vae.config.scaling_factor * latents


@torch.no_grad()
def get_vae_model_and_inputs(
    hf_model_name,
    height,
    width,
    num_channels=4,
    precision="fp16",
    batch_size=1,
    custom_vae_path=None,
):
    # TODO: Switch to amdsharktank implementation.
    dtype = torch_dtypes[precision]
    if dtype == torch.float16:
        custom_vae = "amd-amdshark/sdxl-quant-models"
    else:
        custom_vae = custom_vae_path
    vae_model = VaeModel(hf_model_name, custom_vae=custom_vae).to(dtype=dtype)
    input_image_shape = (batch_size, 3, height, width)
    input_latents_shape = (batch_size, num_channels, height // 8, width // 8)
    encode_args = [
        torch.rand(
            input_image_shape,
            dtype=dtype,
        )
    ]
    decode_args = [
        torch.rand(
            input_latents_shape,
            dtype=dtype,
        ),
    ]
    return vae_model, encode_args, decode_args
