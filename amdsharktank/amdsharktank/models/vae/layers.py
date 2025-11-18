# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from amdsharktank import ops
from amdsharktank.layers import *
from amdsharktank.types import *
from amdsharktank.models.punet.layers import (
    ResnetBlock2D,
    Upsample2D,
    GroupNormLayer,
)
from .config import *


__all__ = ["UNetMidBlock2D", "UpDecoderBlock2D", "AttentionLayer"]

# TODO Remove and integrate with punet AttentionLayer
class AttentionLayer(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        heads: int,  # in_channels // attention_head_dim
        dim_head,
        rescale_output_factor: float,
        eps: float,
        norm_num_groups: int,
        residual_connection: bool,
    ):
        super().__init__(theta)
        self.heads = heads
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection

        if norm_num_groups is not None:
            self.group_norm = GroupNormLayer(
                theta("group_norm"), num_groups=norm_num_groups, eps=eps
            )
        else:
            self.group_norm = None

        self.norm_q = None
        self.norm_k = None

        self.norm_cross = None
        self.to_q = LinearLayer(theta("to_q"))
        self.to_k = LinearLayer(theta("to_k"))
        self.to_v = LinearLayer(theta("to_v"))

        self.added_proj_bias = True
        self.to_out = LinearLayer(theta("to_out")(0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        hidden_states = ops.scaled_dot_product_attention(
            query, key, value, a=attention_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states


class UpDecoderBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        num_layers: int,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        resnet_out_scale_factor: Optional[float],
        resnet_time_scale_shift: str,
        temb_channels: int,
        dropout: float,
        add_upsample: bool,
    ):
        super().__init__(theta)
        resnets = []

        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    theta("resnets")(i),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=resnet_out_scale_factor,
                    time_embedding_norm=resnet_time_scale_shift,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )
            self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(theta("upsamplers")("0"), padding=1)]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UNetMidBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        temb_channels: int,
        dropout: float,
        num_layers: int,
        resnet_eps: float,
        resnet_time_scale_shift: str,
        resnet_act_fn: str,
        resnet_groups: int,
        resnet_pre_norm: bool,
        add_attention: bool,
        attention_head_dim: int,
        output_scale_factor: float,
        attn_groups: Optional[int] = None,
    ):
        super().__init__(theta)

        resnet_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        if resnet_time_scale_shift == "spatial":
            # TODO Implement ResnetBlockCondNorm2d block for spatial time scale shift
            raise AssertionError(f"ResnetBlockCondNorm2d not yet implemented")
        else:
            resnets = [
                ResnetBlock2D(
                    theta("resnets")(0),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    time_embedding_norm=resnet_time_scale_shift,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            ]
        # TODO: loop through num_layers properly. Works for sdxl vae specifically but removed for export reasons
        if add_attention:
            self.attention = AttentionLayer(
                theta("attentions")(0),
                heads=1,
                dim_head=attention_head_dim,
                rescale_output_factor=1.0,
                eps=resnet_eps,
                norm_num_groups=attn_groups,
                residual_connection=True,
            )
        else:
            self.attention = None

        if resnet_time_scale_shift == "spatial":
            # TODO Implement ResnetBlock2D for spatial time scale shift support
            raise AssertionError(
                f"ResnetBlock2D spatial time scale shift not yet implemented"
            )
        else:
            resnets.append(
                ResnetBlock2D(
                    theta("resnets")(1),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    time_embedding_norm=resnet_time_scale_shift,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        if self.attention is not None:
            hidden_states = self.attention(hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states
