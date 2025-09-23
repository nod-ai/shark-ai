# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from safetensors.torch import save_file

from typing import Any, Callable, Dict, List, Optional, Union


# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.bfloat16
)
flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=flow_shift,
)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."


def run_save_t5_goldens(
    pipe,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or pipe._execution_device
    dtype = dtype or pipe.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    inputs_to_save = {"text_input_ids": text_input_ids, "mask": mask}
    save_file(inputs_to_save, "t5_input.safetensors")

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    save_file({"prompt_embeds": prompt_embeds}, "t5_output.safetensors")
    return prompt_embeds


def run_save_transformer_goldens(
    pipe,
    prompt_embeds: torch.Tensor,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    num_frames: int = 20,
    guidance_scale: float = 5.0,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or pipe._execution_device
    latents = pipe.prepare_latents(
        prompt_embeds.shape[0],
        pipe.transformer.config.in_channels,
        height,
        width,
        num_frames,
        torch.float32,
    )
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latent_model_input = latents.to(pipe.transformer.dtype).to(device=device)
    t = timesteps[0]
    timestep = t.expand(latents.shape[0])
    transformer_inputs_for_saving = {
        "hidden_states": latent_model_input,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
    }
    save_file(transformer_inputs_for_saving, "transformer_input.safetensors")
    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        attention_kwargs=None,
        return_dict=False,
    )[0]
    save_file({"noise_pred": noise_pred}, "transformer_output.safetensors")
    latents = pipe.scheduler.step(
        noise_pred.cpu(), t.cpu(), latents.cpu(), return_dict=False
    )[0]
    save_file({"latents": latents}, "scheduler_output.safetensors")
    return latents


def run_save_vae_goldens(
    pipe,
    latents,
):
    latents = latents.to(pipe.vae.dtype)
    pipe.vae.cpu()
    # Vae input processing will be wrapped into the exported VAE compiled module.
    save_file({"latents": latents}, "vae_input.safetensors")
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean

    video = pipe.vae.decode(latents, return_dict=False)[0]
    save_file({"video": video}, "vae_output.safetensors")


if __name__ == "__main__":
    prompt_embeds = run_save_t5_goldens(pipe, prompt)
    latents = run_save_transformer_goldens(
        pipe,
        prompt_embeds,
    )
    run_save_vae_goldens(pipe, latents)


# output = pipe(
#      prompt=prompt,
#      negative_prompt=negative_prompt,
#      height=720,
#      width=1280,
#      num_frames=20,
#      guidance_scale=5.0,
#     ).frames[0]
# export_to_video(output, "output.mp4", fps=16)
