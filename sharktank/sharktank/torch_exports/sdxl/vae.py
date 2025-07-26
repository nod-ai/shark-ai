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
from sharktank.types import Dataset
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.vae.tools.sample_data import get_random_inputs
from sharktank.transforms.dataset import set_float_dtype
from sharktank.tools.import_hf_dataset import import_hf_dataset

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
            # No custom VAE. instantiate from huggingface
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
        elif isinstance(custom_vae, str) and "safetensors" in custom_vae:
            # Custom VAE is a string path to a safetensors file. Load state dict from it.
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
            # Custom VAE is a huggingface model ID
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
def get_sharktank_vae_model_and_inputs(
    model_path,
    height,
    width,
    num_channels=4,
    precision="fp16",
    batch_size=1,
    custom_vae=None,
):
    dtype = torch_dtypes[precision]
    if custom_vae is not None:
        vae_config = os.path.join(model_path, "vae", "config.json")
        vae_model_path = custom_vae
        dataset = import_hf_dataset(
            vae_config,
            [vae_model_path],
        )
    elif os.path.exists(model_path) and "irpa" not in model_path:
        # model_path is a local directory
        vae_config = os.path.join(model_path, "vae", "config.json")
        vae_model_path = os.path.join(model_path, "vae", "vae.safetensors")
        if not os.path.exists(vae_model_path):
            vae_model_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.fp16.safetensors")
        dataset = import_hf_dataset(
            vae_config,
            [vae_model_path],
        )
    elif not os.path.exists(model_path):
        # model_path is a repo id
        vae_config = hf_hub_download(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            revision="main",
            filename="vae/config.json",
        )
        vae_model_path = hf_hub_download(
            repo_id=model_path,
            revision="main",
            filename="vae/vae.safetensors",
        )
        dataset = import_hf_dataset(
            vae_config,
            [vae_model_path],
        )
    else:
        # model_path is an IRPA filepath
        dataset = Dataset.load(model_path, file_type="irpa")

    vae_sharktank_model = VaeDecoderModel.from_dataset(dataset)
        
    input_latents_shape = (batch_size, num_channels, height // 8, width // 8)
    decode_args = [
        torch.rand(
            input_latents_shape,
            dtype=dtype,
        ),
    ]
    return vae_sharktank_model, decode_args

