# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch
import torch.nn as nn

from sharktank import ops
from sharktank.layers import *
from sharktank.types import *

from .config import *
from .layers import *


class Unet2DConditionModel(ThetaLayer):
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "Unet2DConditionModel":
        hp = PunetModelConfig.from_dict(ds.properties["hparams"])
        return cls(hp, ds.root_theta)

    @classmethod
    def from_irpa(cls, irpa_path: str) -> "Unet2DConditionModel":
        ds = Dataset.load(irpa_path, file_type="irpa")
        return cls.from_dataset(ds)

    def __init__(self, config: PunetModelConfig, theta: Theta):
        super().__init__(theta)
        self.hp = config
        # We don't support the full parameterization of the diffusers model, so guard
        # parameters that we require to be their default. This is a tripwire in case
        # if we see a config that requires more support.
        config.assert_default_values(
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
        conv_in_padding = (config.conv_in_kernel - 1) // 2
        self.conv_in = Conv2DLayer(
            theta("conv_in"), padding=(conv_in_padding, conv_in_padding)
        )

        # Down blocks.
        self.down_blocks = nn.ModuleList([])
        for i, down_block_name in enumerate(config.down_block_types):
            down_block_theta = theta("down_blocks", i)
            is_final_block = i == len(config.block_out_channels) - 1
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
        for i, up_block_name in enumerate(config.up_block_types):
            up_block_theta = theta("up_blocks", i)
            is_final_block = i == len(config.block_out_channels) - 1
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
        if config.norm_num_groups is not None:
            self.conv_norm_out = GroupNormLayer(
                theta("conv_norm_out"), num_groups=config.norm_num_groups, eps=config.norm_eps
            )
            self.conv_act = ACTIVATION_FUNCTIONS[config.act_fn]
        conv_out_padding = (config.conv_out_kernel - 1) // 2
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
        add_embeds = torch.cat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
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


class ClassifierFreeGuidanceUnetModel(nn.Module):
    def __init__(self, cond_model: Unet2DConditionModel):
        super().__init__()
        self.cond_model = cond_model
    
    def sample_inputs(self, batch_size: int, function: str, height: int = 1024, width: int = 1024, max_length = 64):
        dtype = torch.float16
        init_batch_dim = 2
        # For classifier free guidance, batch dim multiplies by 2.

        sample = (
            batch_size,
            4,
            height // 8,
            width // 8,
        )
        prompt_embeds_shape = (init_batch_dim * batch_size, max_length, 2048)
        text_embeds_shape = (init_batch_dim * batch_size, 1280)
        time_ids_shape = (init_batch_dim * batch_size, 6)

        standalone_punet_inputs_dict = {
            "sample": torch.rand(sample, dtype=dtype),
            "timestep": torch.zeros(1, dtype=dtype),
            "encoder_hidden_states": torch.rand(prompt_embeds_shape, dtype=dtype),
            "text_embeds": torch.rand(text_embeds_shape, dtype=dtype),
            "time_ids": torch.zeros(time_ids_shape, dtype=dtype),
            "guidance_scale": torch.tensor([7.5], dtype=dtype),
        }
        match function:
            case "forward":
                return None, standalone_punet_inputs_dict
            case _:
                raise NotImplementedError(function)
    
    def forward(
        self, *, sample: torch.Tensor, guidance_scale: torch.Tensor, **cond_kwargs
    ):
        latent_model_input = torch.cat([sample] * 2)
        noise_pred = self.cond_model.forward(sample=latent_model_input, **cond_kwargs)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

class ClassifierFreeGuidanceScheduledUnetModel(nn.Module):
    def __init__(
        self,
        cond_model: Unet2DConditionModel,
        scheduler_model: nn.Module,
        height: int = 1024,
        width: int = 1024,
        max_length: int = 64,
        batch_size: int = 1,
        dtype: str = "fp16",
    ):
        super().__init__()
        self.dtype = dtype
        self.cond_model = cond_model
        self.scheduler = scheduler_model
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_length = max_length
        self.do_classifier_free_guidance = True

        # Preprocess timestep calculations so that we can embed timesteps/sigmas.
        timesteps = [torch.empty((100), dtype=dtype, requires_grad=False)] * 100
        sigmas = [torch.empty((100), dtype=torch.float32, requires_grad=False)] * 100
        for i in range(1, 100):
            self.scheduler.set_timesteps(i)
            timesteps[i] = torch.nn.functional.pad(
                self.scheduler.timesteps.clone().detach(), (0, 100 - i), "constant", 0
            )
            sigmas[i] = torch.nn.functional.pad(
                self.scheduler.sigmas.clone().detach(),
                (0, 100 - (i + 1)),
                "constant",
                0,
            )
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()
        self.sigmas = torch.stack(sigmas, dim=0).clone().detach()

    def sample_inputs(self, function: str):
        dtype = torch.float16

        # For classifier free guidance, batch dim is doubled.
        init_batch_dim = 2

        sample = (
            self.batch_size,
            4,
            self.height // 8,
            self.width // 8,
        )
        prompt_embeds_shape = (init_batch_dim * self.batch_size, self.max_length, 2048)
        text_embeds_shape = (init_batch_dim * self.batch_size, 1280)
        time_ids_shape = (init_batch_dim * self.batch_size, 6)

        init_inputs = (
            torch.rand(sample, dtype=dtype),
            torch.tensor([10], dtype=torch.int64),
        )
        forward_inputs = (
            torch.rand(sample, dtype=dtype),
            torch.rand(prompt_embeds_shape, dtype=dtype),
            torch.rand(text_embeds_shape, dtype=dtype),
            torch.rand(time_ids_shape, dtype=dtype),
            torch.tensor([7.5], dtype=dtype),
            torch.tensor([10], dtype=torch.int64),
            torch.rand(100, dtype=torch.float32),
            torch.rand(100, dtype=torch.float32),
        )
        match function:
            case "forward":
                return forward_inputs, None
            case "initialize":
                return init_inputs, None
            case _:
                raise NotImplementedError(function)
            
    def initialize(self, sample, num_inference_steps):
        height = self.height
        width = self.width
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1).type(self.dtype)
        max_sigma = self.sigmas[num_inference_steps].max()
        init_noise_sigma = (max_sigma**2 + 1) ** 0.5
        sample = sample * init_noise_sigma
        return (
            sample.type(self.dtype),
            add_time_ids,
            self.timesteps[num_inference_steps].squeeze(0),
            self.sigmas[num_inference_steps].squeeze(0),
        )

    def scale_model_input(self, sample, i, timesteps, sigmas):
        sigma = sigmas[i]
        next_sigma = sigmas[i + 1]
        t = timesteps[i]
        latent_model_input = sample / ((sigma**2 + 1) ** 0.5)
        return (
            latent_model_input.type(self.dtype),
            t.type(self.dtype),
            sigma.type(self.dtype),
            next_sigma.type(self.dtype),
        )

    def step(self, noise_pred, sample, sigma, next_sigma):
        sample = sample.to(torch.float32)
        gamma = 0.0
        noise_pred = noise_pred.to(torch.float32)
        sigma_hat = sigma * (gamma + 1)
        pred_original_sample = sample - sigma_hat * noise_pred
        deriv = (sample - pred_original_sample) / sigma_hat
        dt = next_sigma - sigma_hat
        prev_sample = sample + deriv * dt
        return prev_sample.type(self.dtype)

    def forward(
        self,
        latents,
        prompt_embeds,
        text_embeds,
        time_ids,
        guidance_scale,
        step_index,
        timesteps,
        sigmas,
    ):
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input, t, sigma, next_sigma = self.scale_model_input(
            latent_model_input, step_index, timesteps, sigmas
        )

        noise_pred = self.cond_model.forward(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            time_ids=time_ids,
            text_embeds=text_embeds,
        )

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        sample = self.step(noise_pred, latents, sigma, next_sigma)
        return sample
