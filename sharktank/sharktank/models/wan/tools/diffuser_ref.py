# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch
from diffusers import WanTransformer3DModel
from einops import rearrange
from collections import OrderedDict
import math

from ..wan import WanModel


def convert_wan_transformer_to_hugging_face(model: WanModel) -> WanTransformer3DModel:
    hf_model = WanTransformer3DModel(**model.hp.to_hugging_face_config())
    state_dict = {k: v.as_torch() for k, v in model.theta.flatten().items()}
    hf_model.load_state_dict(state_dict)
    return hf_model


class WanModel(torch.nn.Module):
    def __init__(
        self,
        hf_model: WanTransformer3DModel | str,
        height=512,
        width=512,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.model = None
        self.height = height
        self.width = width
        if isinstance(hf_model, WanTransformer3DModel):
            self.model = hf_model
        elif isinstance(hf_model, str):
            self.model = WanTransformer3DModel.from_pretrained(
                hf_model, subfolder="transformer"
            ).to(dtype=dtype)

    def forward(self, x, t, context) -> torch.Tensor:
        return self.model.forward(x, t, context)


def convert_wan_transformer_input_for_hugging_face_model(
    x: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("hidden_states", x),
            ("encoder_hidden_states", context),
            ("timestep", t),
        ]
    )
