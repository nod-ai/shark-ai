# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch
from diffusers import AutoencoderKL


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
        elif "safetensors" in custom_vae:
            custom_vae = safetensors.torch.load_file(custom_vae)
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            self.vae.load_state_dict(custom_vae)
        elif not isinstance(custom_vae, dict):
            try:
                # custom HF repo with no vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                )
            except:
                # some larger repo with vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                    subfolder="vae",
                )

    def decode(self, inp):
        # The reference vae decode does not do scaling and leaves it for the sdxl pipeline. We integrate it into vae for pipeline performance so using the hardcoded values from the config.json here
        img = 1 / 0.13025 * inp
        x = self.vae.decode(img, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)


def run_torch_vae(hf_model_name, example_input):

    vae_model = VaeModel(hf_model_name)
    return vae_model.decode(example_input)
