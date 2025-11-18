# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import transformers
import torch
import functools

from amdsharktank.layers.configs.llm_configs import T5Config
from amdsharktank.types import Theta, torch_module_to_theta
from amdsharktank.transforms.dataset import set_float_dtype
from .t5 import T5Encoder


def get_t5_encoder_toy_config() -> T5Config:
    return T5Config(
        vocab_size=5,
        context_length=13,
        d_model=17,
        d_kv=19,
        d_ff=23,
        num_layers=2,
        num_decoder_layers=0,
        num_heads=3,
        relative_attention_num_buckets=7,
        relative_attention_max_distance=11,
        layer_norm_epsilon=1e-2,
    )


def covert_t5_encoder_to_hugging_face(model: T5Encoder) -> transformers.T5EncoderModel:
    hf_config = model.config.to_hugging_face_config()
    with transformers.modeling_utils.no_init_weights():
        hf_model = transformers.T5EncoderModel(hf_config)
    state_dict = {k: v.as_torch() for k, v in model.theta.flatten().items()}
    state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    hf_model.load_state_dict(state_dict)
    return hf_model


def make_t5_encoder_random_theta(config: T5Config, /, *, dtype: torch.dtype) -> Theta:
    hf_config = config.to_hugging_face_config()
    hf_model = transformers.T5EncoderModel(hf_config)
    theta = torch_module_to_theta(hf_model)
    return theta.transform(functools.partial(set_float_dtype, dtype=dtype))
