# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn

from amdsharktank import ops
from amdsharktank.layers import *
from amdsharktank.types import *

from .config import *
from .layers import *


class Unet2DConditionModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "Unet2DConditionModel":
        hp = HParams.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    def __init__(self, hp: HParams, theta: Theta):
        super().__init__(theta)
        self.hp = hp
        # We don't support the full parameterization of the diffusers model, so guard
        # parameters that we require to be their default. This is a tripwire in case
        # if we see a config that requires more support.
        hp.assert_default_values(
            [
                "addition_embed_type",
                "center_input_sample",
                "class_embed_type",
                "class_embeddings_concat",
                "dual_cross_attention",
                "encoder_hid_dim",
                "encoder_hid_dim_type",
                "freq_shift",
                "mid_block_scale_factor",
                "only_cross_attention",
                "time_embedding_act_fn",
                "time_embedding_dim",
                "time_embedding_type",
                "timestep_post_act",
                "upcast_attention",
            ]
        )
        self._setup_timestep_embedding()
        self._setup_addition_embedding()

        # Input convolution.
        conv_in_padding = (hp.conv_in_kernel - 1) // 2
        self.conv_in = Conv2DLayer(
            theta("conv_in"), padding=(conv_in_padding, conv_in_padding)
        )

        # Down blocks.
        self.down_blocks = nn.ModuleList([])
        for i, down_block_name in enumerate(hp.down_block_types):
            down_block_theta = theta("down_blocks", i)
            is_final_block = i == len(hp.block_out_channels) - 1
            self.down_blocks.append(
                self._create_down_block(
                    i,
                    down_block_theta,
                    down_block_name,
                    is_final_block=is_final_block,
                )
            )

        # Mid block.
        self.mid_block = self._create_mid_block(theta("mid_block"))

        # Up blocks.
        self.up_blocks = nn.ModuleList([])
        for i, up_block_name in enumerate(hp.up_block_types):
            up_block_theta = theta("up_blocks", i)
            is_final_block = i == len(hp.block_out_channels) - 1
            self.up_blocks.append(
                self._create_up_block(
                    i,
                    up_block_theta,
                    up_block_name,
                    is_final_block=is_final_block,
                )
            )

        # Output.
        self.conv_norm_out = None
        self.conv_act = None
        if hp.norm_num_groups is not None:
            self.conv_norm_out = GroupNormLayer(
                theta("conv_norm_out"), num_groups=hp.norm_num_groups, eps=hp.norm_eps
            )
            self.conv_act = ACTIVATION_FUNCTIONS[hp.act_fn]
        conv_out_padding = (hp.conv_out_kernel - 1) // 2
        self.conv_out = Conv2DLayer(
            theta("conv_out"), padding=(conv_out_padding, conv_out_padding)
        )

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ):
        """
        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            text_embeds: Additional embedding.
            time_ids: Additional embedding.
        """
        # Invariants.
        torch._check(len(timestep.shape) == 1 and timestep.shape[0] == 1)
        # TODO: Verify on the fly upsampling is not needed (num_upsamplers != 0).
        act_dtype = sample.dtype
        bs, *_ = sample.shape
        self.trace_tensor(
            "inputs",
            {
                "sample": sample,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
        )

        # 0. Center input if necessary.
        assert not self.hp.center_input_sample, "NYI: Center input sample"

        # 1. Embeddings.
        # 1a. Time embedding.
        # Broadcast the timestep to the batch size ([1] -> [bs]), apply projection
        # and cast.
        t_emb = self.time_proj(timestep.expand(bs)).to(dtype=act_dtype)
        emb = self.time_embedding(t_emb)

        # 1b. Aug embedding of text_embeds, time_ids
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = ops.cat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
        aug_embed = self.add_embedding(add_embeds)
        emb = emb + aug_embed
        self.trace_tensor("emb", emb)

        # 2. Pre-process.
        sample = self.conv_in(sample)
        self.trace_tensor("preprocess", sample)

        # 3. Down.
        down_block_res_samples = (sample,)
        for i, down_block in enumerate(self.down_blocks):
            sample, res_samples = down_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                encoder_attention_mask=None,
            )
            down_block_res_samples += res_samples
            self.trace_tensor(f"down_block_{i}", sample)

        # 4. Mid.
        sample, _ = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            encoder_attention_mask=None,
        )
        self.trace_tensor("mid_block", sample)

        # 5. Up.
        for i, up_block in enumerate(self.up_blocks):
            # Rotate res samples in LIFO order.
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]

            sample, _ = up_block(
                hidden_states=sample,
                res_hidden_states_tuple=res_samples,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                encoder_attention_mask=None,
            )
            self.trace_tensor(f"up_block_{i}", sample)

        # 6. Post-process.
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = ops.elementwise(self.conv_act, sample)
        sample = self.conv_out(sample)
        self.trace_tensor(f"output", sample)
        return sample

    def _create_down_block(
        self, i: int, down_block_theta: Theta, type_name: str, is_final_block: bool
    ) -> nn.Module:
        hp = self.hp
        if type_name == "DownBlock2D":
            return UpDownBlock2D(
                down_block_theta,
                num_layers=hp.layers_per_block[i],
                add_downsample=not is_final_block,
                downsample_padding=hp.downsample_padding,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=(
                    None
                    if hp.resnet_out_scale_factor == 1.0
                    else hp.resnet_out_scale_factor
                ),
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                temb_channels=self.time_embed_dim,
            )
        elif type_name == "CrossAttnDownBlock2D":
            return CrossAttnUpDownBlock2D(
                down_block_theta,
                num_layers=hp.layers_per_block[i],
                transformer_layers_per_block=hp.transformer_layers_per_block[i],
                num_attention_heads=hp.num_attention_heads[i],
                cross_attention_dim=hp.cross_attention_dim[i],
                temb_channels=self.time_embed_dim,
                add_downsample=not is_final_block,
                downsample_padding=hp.downsample_padding,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=(
                    None
                    if hp.resnet_out_scale_factor == 1.0
                    else hp.resnet_out_scale_factor
                ),
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                use_linear_projection=hp.use_linear_projection,
            )
        raise ValueError(f"Unhandled down_block_type: {type_name}")

    def _create_mid_block(self, mid_block_theta: Theta) -> nn.Module:
        hp = self.hp
        if hp.mid_block_type == "UNetMidBlock2DCrossAttn":
            return CrossAttnUpDownBlock2D(
                mid_block_theta,
                num_layers=1,
                num_prefix_resnets=1,  # Mid block always has an additional resnet.
                transformer_layers_per_block=hp.transformer_layers_per_block[-1],
                num_attention_heads=hp.num_attention_heads[-1],
                cross_attention_dim=hp.cross_attention_dim[-1],
                temb_channels=self.time_embed_dim,
                add_downsample=False,
                downsample_padding=hp.downsample_padding,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=(
                    None
                    if hp.resnet_out_scale_factor == 1.0
                    else hp.resnet_out_scale_factor
                ),
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                use_linear_projection=hp.use_linear_projection,
            )

        raise ValueError("Unhandled mid_block_type: {type_name}")

    def _create_up_block(
        self, i: int, up_block_theta: Theta, type_name: str, is_final_block: bool
    ) -> nn.Module:
        def r(s):
            return list(reversed(s))

        hp = self.hp
        if type_name == "UpBlock2D":
            return UpDownBlock2D(
                up_block_theta,
                num_layers=r(hp.layers_per_block)[i] + 1,
                add_upsample=not is_final_block,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=(
                    None
                    if hp.resnet_out_scale_factor == 1.0
                    else hp.resnet_out_scale_factor
                ),
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                temb_channels=self.time_embed_dim,
            )
        elif type_name == "CrossAttnUpBlock2D":
            return CrossAttnUpDownBlock2D(
                up_block_theta,
                num_layers=r(hp.layers_per_block)[i] + 1,
                transformer_layers_per_block=r(hp.transformer_layers_per_block)[i],
                num_attention_heads=r(hp.num_attention_heads)[i],
                cross_attention_dim=r(hp.cross_attention_dim)[i],
                temb_channels=self.time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=hp.norm_eps,
                resnet_act_fn=hp.act_fn,
                resnet_groups=hp.norm_num_groups,
                resnet_out_scale_factor=(
                    None
                    if hp.resnet_out_scale_factor == 1.0
                    else hp.resnet_out_scale_factor
                ),
                resnet_time_scale_shift=hp.resnet_time_scale_shift,
                dropout=hp.dropout,
                use_linear_projection=hp.use_linear_projection,
            )
        raise ValueError(f"Unhandled up_block_type: {type_name}")

    def _setup_timestep_embedding(self):
        hp = self.hp
        assert hp.time_embedding_type == "positional", "NYI"
        self.time_embed_dim = time_embed_dim = hp.block_out_channels[0] * 2
        timestep_input_dim = hp.block_out_channels[0]
        self.time_proj = TimestepProjection(
            hp.block_out_channels[0], flip_sin_to_cos=hp.flip_sin_to_cos
        )
        self.time_embedding = TimestepEmbedding(
            self.theta("time_embedding"),
            timestep_input_dim,
            time_embed_dim,
            act_fn=hp.act_fn,
        )

    def _setup_addition_embedding(self):
        hp = self.hp
        assert hp.addition_embed_type == "text_time", "NYI"
        self.add_time_proj = TimestepProjection(
            hp.addition_time_embed_dim,
            downscale_freq_shift=hp.freq_shift,
            flip_sin_to_cos=hp.flip_sin_to_cos,
        )
        self.add_embedding = TimestepEmbedding(
            self.theta("add_embedding"),
            in_channels=hp.projection_class_embeddings_input_dim,
            time_embed_dim=hp.time_embedding_dim,
            act_fn=hp.act_fn,
        )


class ClassifierFreeGuidanceUnetModel(torch.nn.Module):
    def __init__(self, cond_model: Unet2DConditionModel):
        super().__init__()
        self.cond_model = cond_model

    def forward(
        self, *, sample: torch.Tensor, guidance_scale: torch.Tensor, **cond_kwargs
    ):
        latent_model_input = ops.cat([sample] * 2)
        noise_pred = self.cond_model.forward(sample=latent_model_input, **cond_kwargs)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred
