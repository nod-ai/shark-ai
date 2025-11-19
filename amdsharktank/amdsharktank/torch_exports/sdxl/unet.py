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
from huggingface_hub import hf_hub_download

import torch

from .scheduler import get_scheduler, SchedulingModel


class ScheduledUnetModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        unet_module,
        scheduler_module,
        height,
        width,
        batch_size,
        dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.cond_model = unet_module
        self.scheduler = scheduler_module
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.do_classifier_free_guidance = True
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


@torch.no_grad()
def get_scheduled_unet_model_and_inputs(
    hf_model_name,
    height,
    width,
    max_length,
    precision,
    batch_size,
    external_weight_path,
    quant_path,
    scheduler_config_path=None,
):
    if quant_path is not None and os.path.exists(quant_path):
        quant_paths = {
            "config": f"{quant_path}/config.json",
            "params": f"{quant_path}/params.safetensors",
            "quant_params": f"{quant_path}/quant_params.json",
        }
    else:
        quant_paths = None
    if precision == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16
    unet = get_punet_model(
        hf_model_name,
        external_weight_path,
        quant_paths,
        precision,
    )
    if not scheduler_config_path:
        scheduler_config_path = hf_model_name
    raw_scheduler = get_scheduler(scheduler_config_path, "EulerDiscrete")
    model = ScheduledUnetModel(
        hf_model_name,
        unet,
        raw_scheduler,
        height,
        width,
        batch_size,
        dtype,
    )

    init_batch_dim = 2
    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    prompt_embeds_shape = (init_batch_dim * batch_size, max_length, 2048)
    text_embeds_shape = (init_batch_dim * batch_size, 1280)
    time_ids_shape = (init_batch_dim * batch_size, 6)

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
    return model, init_inputs, forward_inputs


@torch.no_grad()
def get_punet_model_and_inputs(
    hf_model_name,
    height,
    width,
    max_length,
    precision,
    batch_size,
    external_weight_path,
    quant_path=None,
    scheduler_config_path=None,
):
    from amdsharktank.models.punet.model import (
        ClassifierFreeGuidanceUnetModel as CFGPunet,
    )

    if quant_path is not None and os.path.exists(quant_path):
        quant_paths = {
            "config": f"{quant_path}/config.json",
            "params": f"{quant_path}/params.safetensors",
            "quant_params": f"{quant_path}/quant_params.json",
        }
    else:
        quant_paths = None

    if precision == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16
    mod = get_punet_model(
        hf_model_name,
        external_weight_path,
        quant_paths,
        precision,
    )
    model = CFGPunet(mod)

    init_batch_dim = 2
    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    prompt_embeds_shape = (init_batch_dim * batch_size, max_length, 2048)
    text_embeds_shape = (init_batch_dim * batch_size, 1280)
    time_ids_shape = (init_batch_dim * batch_size, 6)

    standalone_unet_inputs = {
        "sample": torch.rand(sample, dtype=dtype),
        "timestep": torch.zeros(1, dtype=dtype),
        "encoder_hidden_states": torch.rand(prompt_embeds_shape, dtype=dtype),
        "text_embeds": torch.rand(text_embeds_shape, dtype=dtype),
        "time_ids": torch.zeros(time_ids_shape, dtype=dtype),
        "guidance_scale": torch.tensor([7.5], dtype=dtype),
    }
    return model, None, standalone_unet_inputs


def get_punet_model(hf_model_name, external_weight_path, quant_paths, precision="i8"):
    from amdsharktank.models.punet.model import (
        Unet2DConditionModel as amdsharktank_unet2d,
    )
    from amdsharktank.utils import cli, hf_datasets
    from amdsharktank.tools.import_hf_dataset import import_hf_dataset

    if precision in ["fp8", "f8"]:
        repo_id = "amd-amdshark/sdxl-quant-models"
        subfolder = "unet/int8"
        revision = "a31d1b1cba96f0da388da348bcaee197a073d451"
    elif precision == "fp8_ocp":
        repo_id = "amd-amdshark/sdxl-quant-fp8"
        subfolder = "unet_int8_sdpa_fp8_ocp"
        revision = "e6e3c031e6598665ca317b80c3b627c186ca08e7"
    else:
        repo_id = "amd-amdshark/sdxl-quant-int8"
        subfolder = "mi300_all_sym_8_step14_fp32"
        revision = "efda8afb35fd72c1769e02370b320b1011622958"

    # TODO (monorimet): work through issues with pure fp16 punet export. Currently int8 with fp8/fp8_ocp/fp16 sdpa are supported.

    def download(filename):
        return hf_hub_download(
            repo_id=repo_id, subfolder=subfolder, filename=filename, revision=revision
        )

    if quant_paths and quant_paths["config"] and os.path.exists(quant_paths["config"]):
        results = {
            "config.json": quant_paths["config"],
        }
    else:
        try:
            results = {
                "config.json": download("config.json"),
            }
        except:
            # Fallback to original model config file.
            results = {
                "config.json": hf_hub_download(
                    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                    subfolder="unet",
                    filename="config.json",
                )
            }
    if quant_paths and quant_paths["params"] and os.path.exists(quant_paths["params"]):
        results["params.safetensors"] = quant_paths["params"]
    else:
        results["params.safetensors"] = download("params.safetensors")

    output_dir = os.path.dirname(external_weight_path)

    if (
        quant_paths
        and quant_paths["quant_params"]
        and os.path.exists(quant_paths["quant_params"])
    ):
        results["quant_params.json"] = quant_paths["quant_params"]
    else:
        results["quant_params.json"] = download("quant_params.json")
    ds_filename = os.path.basename(external_weight_path)
    output_path = os.path.join(output_dir, ds_filename)
    ds = get_punet_dataset(
        results["config.json"],
        results["params.safetensors"],
        output_path,
        results["quant_params.json"],
    )

    cond_unet = amdsharktank_unet2d.from_dataset(ds)
    return cond_unet


def get_punet_dataset(
    config_json_path,
    params_path,
    output_path,
    quant_params_path=None,
):
    from amdsharktank.models.punet.tools import import_brevitas_dataset

    ds_import_args = [
        f"--config-json={config_json_path}",
        f"--params={params_path}",
        f"--output-irpa-file={output_path}",
    ]
    if quant_params_path:
        ds_import_args.extend([f"--quant-params={quant_params_path}"])
    import_brevitas_dataset.main(ds_import_args)
    return import_brevitas_dataset.Dataset.load(output_path)
