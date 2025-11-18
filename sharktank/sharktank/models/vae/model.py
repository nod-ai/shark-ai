# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from einops import rearrange
import math

import torch
import torch.nn as nn

from amdsharktank.layers import *
from amdsharktank.types import *
from .config import *
from .layers import *
from amdsharktank.models.punet.layers import GroupNormLayer


class VaeDecoderModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "VaeDecoderModel":
        hp = HParams.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    def __init__(self, hp: HParams, theta: Theta):
        super().__init__(theta)
        self.hp = hp

        # input conv
        if hp.use_post_quant_conv:
            self.post_quant_conv = Conv2DLayer(theta("post_quant_conv"), padding=(0, 0))
        self.conv_in = Conv2DLayer(theta("decoder")("conv_in"), padding=(1, 1))
        # Mid
        self.mid_block = self._create_mid_block(theta("decoder")("mid_block"))
        # up
        self.up_blocks = nn.ModuleList([])
        self.upscale_dtype = unbox_tensor(
            theta("decoder")("up_blocks")(0)("resnets")(0)("conv1")("weight")
        ).dtype
        for i, up_block_name in enumerate(hp.up_block_types):
            up_block_theta = theta("decoder")("up_blocks")(i)
            is_final_block = i == len(hp.block_out_channels) - 1
            self.up_blocks.append(
                self._create_up_block(
                    up_block_theta,
                    up_block_name,
                    is_final_block=is_final_block,
                )
            )
        # TODO add spatial norm type support
        self.conv_norm_out = GroupNormLayer(
            theta("decoder")("conv_norm_out"), num_groups=hp.norm_num_groups, eps=1e-6
        )

        self.conv_act = nn.SiLU()
        self.conv_out = Conv2DLayer(theta("decoder")("conv_out"), padding=(1, 1))

        self.sample_size = hp.sample_size
        if isinstance(self.sample_size, int):
            # (height, width)
            self.sample_size = (self.sample_size, self.sample_size)

    def forward(
        self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The forward method of the 'Decoder' class
        Args:
            sample ('torch.Tensor') input latents of shape (batch_size, num_channels, height, width)

        """
        self.trace_tensor(
            "inputs",
            {
                "sample": sample,
                "latent_embeds": latent_embeds,
            },
        )
        if not self.hp.use_post_quant_conv:
            sample = rearrange(
                sample,
                "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                h=math.ceil(self.sample_size[0] / 16),
                w=math.ceil(self.sample_size[1] / 16),
                ph=2,
                pw=2,
            )

        sample = sample / self.hp.scaling_factor + self.hp.shift_factor

        if self.hp.use_post_quant_conv:
            sample = self.post_quant_conv(sample)

        sample = self.conv_in(sample)
        self.trace_tensor("conv_in", sample)
        # TODO add training and gradient checkpointing support
        sample = self.mid_block(sample, latent_embeds)
        self.trace_tensor("mid_block", sample)

        sample = sample.to(self.upscale_dtype)
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)

        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not self.hp.use_post_quant_conv:
            sample = sample.clamp(-1, 1)
        else:
            sample = (sample / 2 + 0.5).clamp(0, 1)
        return sample

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_out.weight.dtype

    def _create_mid_block(self, mid_block_theta: Theta) -> nn.Module:
        hp = self.hp
        return UNetMidBlock2D(
            mid_block_theta,
            temb_channels=None,
            dropout=0.0,
            num_layers=hp.layers_per_block,
            resnet_eps=1e-6,
            resnet_act_fn="swish",
            resnet_groups=hp.norm_num_groups,
            attn_groups=hp.norm_num_groups,
            resnet_pre_norm=True,
            add_attention=True,
            attention_head_dim=hp.block_out_channels[-1],
            output_scale_factor=1.0,
            resnet_time_scale_shift="default",
        )

    def _create_up_block(
        self, up_block_theta: Theta, type_name: str, is_final_block: bool
    ) -> nn.Module:
        hp = self.hp
        if type_name == "UpDecoderBlock2D":
            return UpDecoderBlock2D(
                up_block_theta,
                num_layers=hp.layers_per_block + 1,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_time_scale_shift="default",
                temb_channels=None,
                dropout=0.0,
                resnet_out_scale_factor=None,
            )
