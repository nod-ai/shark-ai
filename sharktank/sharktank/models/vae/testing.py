# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import torch

from .config import HParams
from amdsharktank.types import Theta, torch_module_to_theta
from amdsharktank.transforms.dataset import set_float_dtype


def get_toy_vae_decoder_config() -> HParams:
    # This config causes compiler error
    # https://github.com/iree-org/iree/issues/20307
    # norm_num_groups = 3
    # return HParams(
    #     block_out_channels=[norm_num_groups * 7, norm_num_groups * 11],
    #     in_channels=5,
    #     out_channels=17,
    #     up_block_types=[
    #         "UpDecoderBlock2D",
    #         "UpDecoderBlock2D",
    #     ],
    #     norm_num_groups=norm_num_groups,
    #     latent_channels=13,
    #     layers_per_block=2,
    #     shift_factor=0.1234,
    #     use_post_quant_conv=False,
    #     sample_size=[23, 29],
    # )

    norm_num_groups = 2
    return HParams(
        block_out_channels=[norm_num_groups * 3, norm_num_groups * 4],
        in_channels=5,
        out_channels=6,
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        norm_num_groups=norm_num_groups,
        latent_channels=7,
        layers_per_block=8,
        shift_factor=0.1234,
        use_post_quant_conv=False,
        sample_size=[128, 128],
    )


def make_vae_decoder_random_theta(config: HParams, /, *, dtype: torch.dtype) -> Theta:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    hf_config = config.to_hugging_face_config()
    hf_model = AutoencoderKL(**hf_config)
    theta = torch_module_to_theta(hf_model)
    return theta.transform(functools.partial(set_float_dtype, dtype=dtype))
