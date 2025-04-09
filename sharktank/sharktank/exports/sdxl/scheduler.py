# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.aot import *
from diffusers import (
    EulerDiscreteScheduler,
)


class SchedulingModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        scheduler,
        height,
        width,
        batch_size,
        dtype,
    ):
        super().__init__()
        # For now, assumes SDXL implementation. May not need parametrization for other models,
        # but keeping hf_model_name in case.
        self.model = scheduler
        self.height = height
        self.width = width
        self.is_sd3 = False
        if "stable-diffusion-3" in hf_model_name:
            self.is_sd3 = True
        self.batch_size = batch_size
        # Whether this will be used with CFG-enabled pipeline.
        self.do_classifier_free_guidance = True
        # Prefetch a list of timesteps and sigmas for steps 1-100.
        timesteps = [torch.empty((100), dtype=dtype, requires_grad=False)] * 100
        sigmas = [torch.empty((100), dtype=torch.float32, requires_grad=False)] * 100
        for i in range(1, 100):
            self.model.set_timesteps(i)
            timesteps[i] = torch.nn.functional.pad(
                self.model.timesteps.clone().detach(), (0, 100 - i), "constant", 0
            )
            sigmas[i] = torch.nn.functional.pad(
                self.model.sigmas.clone().detach(), (0, 100 - (i + 1)), "constant", 0
            )
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()
        self.sigmas = torch.stack(sigmas, dim=0).clone().detach()
        self.model.is_scale_input_called = True
        self.dtype = dtype

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
        self.model.is_scale_input_called = True
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


def get_scheduler(model_id, scheduler_id):
    if scheduler_id in SCHEDULER_MAP.keys():
        scheduler = SCHEDULER_MAP[scheduler_id].from_pretrained(
            model_id, subfolder="scheduler"
        )
    elif all(x in scheduler_id for x in ["DPMSolverMultistep", "++"]):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler", algorithm_type="dpmsolver++"
        )
    else:
        raise ValueError(f"Scheduler {scheduler_id} not found.")
    if "Karras" in scheduler_id:
        scheduler.config.use_karras_sigmas = True

    return scheduler


SCHEDULER_MAP = {
    "EulerDiscrete": EulerDiscreteScheduler,
}

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_scheduler_model_and_inputs(
    hf_model_name,
    batch_size,
    height,
    width,
    precision,
    scheduler_id="EulerDiscrete",
):
    dtype = torch_dtypes[precision]
    raw_scheduler = get_scheduler(hf_model_name, scheduler_id)
    scheduler = SchedulingModel(
        hf_model_name, raw_scheduler, height, width, batch_size, dtype
    )
    init_in, prep_in, step_in = get_sample_sched_inputs(
        batch_size, height, width, dtype
    )
    return scheduler, init_in, prep_in, step_in


def get_sample_sched_inputs(batch_size, height, width, dtype):
    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    noise_pred_shape = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    init_args = (
        torch.rand(sample, dtype=dtype),
        torch.tensor([10], dtype=torch.int64),
    )
    prep_args = (
        torch.rand(sample, dtype=dtype),
        torch.tensor([10], dtype=torch.int64),
        torch.rand(100, dtype=torch.float32),
        torch.rand(100, dtype=torch.float32),
    )
    step_args = [
        torch.rand(noise_pred_shape, dtype=dtype),
        torch.rand(sample, dtype=dtype),
        torch.rand(1, dtype=dtype),
        torch.rand(1, dtype=dtype),
    ]
    return init_args, prep_args, step_args
