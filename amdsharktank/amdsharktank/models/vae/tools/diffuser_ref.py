# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch
from diffusers import AutoencoderKL
from einops import rearrange
import math

from ..model import VaeDecoderModel


def convert_vae_decoder_to_hugging_face(model: VaeDecoderModel) -> AutoencoderKL:
    hf_model = AutoencoderKL(**model.hp.to_hugging_face_config())
    state_dict = {k: v.as_torch() for k, v in model.theta.flatten().items()}
    hf_model.load_state_dict(state_dict)
    return hf_model


class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model: AutoencoderKL | str,
        custom_vae="",
        height=1024,
        width=1024,
        flux=False,
    ):
        super().__init__()
        self.vae = None
        self.height = height
        self.width = width
        self.flux = flux
        if isinstance(hf_model, AutoencoderKL):
            self.vae = hf_model
        elif custom_vae in ["", None]:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model,
                subfolder="vae",
            )
        elif "safetensors" in custom_vae:
            custom_vae = safetensors.torch.load_file(custom_vae)
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model,
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
        self.shift_factor = (
            0.0
            if self.vae.config.shift_factor is None
            else self.vae.config.shift_factor
        )

    def decode(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.to(dtype=self.dtype)
        if self.flux:
            assert self.vae.post_quant_conv is None  # No post quant
            inp = rearrange(
                inp,
                "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                h=math.ceil(self.height / 16),
                w=math.ceil(self.width / 16),
                ph=2,
                pw=2,
            )
        img = inp / self.vae.config.scaling_factor + self.shift_factor
        x = self.vae.decode(img, return_dict=False)[0]
        if self.flux:
            return x.clamp(-1, 1)
        else:
            return (x / 2 + 0.5).clamp(0, 1)

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.decoder.conv_out.weight.dtype


def run_torch_vae(
    hf_model_name,
    example_input,
    height=1024,
    width=1024,
    flux=False,
    dtype: torch.dtype | None = None,
):
    vae_model = VaeModel(hf_model_name, height=height, width=width, flux=flux)
    if dtype is not None:
        vae_model = vae_model.to(dtype)
    return vae_model.decode(example_input)
