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

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPTextConfig,
)


class PromptEncoderModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        precision,
        batch_size=1,
        batch_input=False,
    ):
        super().__init__()
        self.torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        config_1 = CLIPTextConfig.from_pretrained(
            hf_model_name,
            subfolder="text_encoder",
        )
        config_1._attn_implementation = "eager"
        config_2 = CLIPTextConfig.from_pretrained(
            hf_model_name,
            subfolder="text_encoder_2",
        )
        config_2._attn_implementation = "eager"
        self.text_encoder_model_1 = CLIPTextModel.from_pretrained(
            hf_model_name,
            config=config_1,
            subfolder="text_encoder",
        )
        self.text_encoder_model_2 = CLIPTextModelWithProjection.from_pretrained(
            hf_model_name,
            config=config_2,
            subfolder="text_encoder_2",
        )
        self.do_classifier_free_guidance = True
        self.batch_size = batch_size
        self.batch_input = batch_input

    def forward(
        self, text_input_ids_1, text_input_ids_2, uncond_input_ids_1, uncond_input_ids_2
    ):
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder_model_1(
                text_input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds_2 = self.text_encoder_model_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            neg_prompt_embeds_1 = self.text_encoder_model_1(
                uncond_input_ids_1,
                output_hidden_states=True,
            )
            neg_prompt_embeds_2 = self.text_encoder_model_2(
                uncond_input_ids_2,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_2[0]
            neg_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            prompt_embeds_list = [
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ]
            neg_prompt_embeds_list = [
                neg_prompt_embeds_1.hidden_states[-2],
                neg_prompt_embeds_2.hidden_states[-2],
            ]

            prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
            neg_prompt_embeds = torch.cat(neg_prompt_embeds_list, dim=-1)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
                bs_embed * 1, -1
            )
            if not self.batch_input:
                prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
            add_text_embeds = pooled_prompt_embeds
            if not self.batch_input:
                add_text_embeds = add_text_embeds.repeat(self.batch_size, 1)
            if self.do_classifier_free_guidance:
                if not self.batch_input:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(
                        1, 1
                    ).view(1, -1)
                neg_prompt_embeds = neg_prompt_embeds.repeat(1, 1, 1)
                neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * 1, seq_len, -1)
                if not self.batch_input:
                    neg_prompt_embeds = neg_prompt_embeds.repeat(self.batch_size, 1, 1)
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                if not self.batch_input:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(
                        self.batch_size, 1
                    )
                add_text_embeds = torch.cat(
                    [neg_pooled_prompt_embeds, add_text_embeds], dim=0
                )
            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds


@torch.no_grad()
def get_clip_model_and_inputs(
    hf_model_name,
    max_length=64,
    precision="fp16",
    batch_size=1,
    batch_input=True,
):
    # TODO: Switch to amdsharktank CLIP implementation.
    prompt_encoder_module = PromptEncoderModel(
        hf_model_name,
        precision,
        batch_size=batch_size,
        batch_input=batch_input,
    )

    input_batchsize = 1
    if batch_input:
        input_batchsize = batch_size

    if precision == "fp16":
        prompt_encoder_module = prompt_encoder_module.half()

    example_inputs = {
        "text_input_ids_1": torch.ones(
            (input_batchsize, max_length), dtype=torch.int64
        ),
        "text_input_ids_2": torch.ones(
            (input_batchsize, max_length), dtype=torch.int64
        ),
        "uncond_input_ids_1": torch.ones(
            (input_batchsize, max_length), dtype=torch.int64
        ),
        "uncond_input_ids_2": torch.ones(
            (input_batchsize, max_length), dtype=torch.int64
        ),
    }
    return prompt_encoder_module, example_inputs
