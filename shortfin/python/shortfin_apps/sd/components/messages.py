# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import logging

import shortfin as sf
import shortfin.array as sfnp
import numpy as np

from .io_struct import GenerateReqInput

logger = logging.getLogger("shortfin-sd.messages")


class InferencePhase(Enum):
    # Tokenize prompt, negative prompt and get latents, timesteps, time ids, guidance scale as device arrays
    PREPARE = 1
    # Run CLIP to encode tokenized prompts into text embeddings
    ENCODE = 2
    # Run UNet to denoise the random sample
    DENOISE = 3
    # Run VAE to decode the denoised latents into an image.
    DECODE = 4
    # Postprocess VAE outputs.
    POSTPROCESS = 5


class InferenceExecRequest(sf.Message):
    """
    Generalized request passed for an individual phase of image generation.

    Used for individual image requests. Bundled as lists by the batcher for inference processes,
    and inputs joined for programs with bs>1.

    Inference execution processes are responsible for writing their outputs directly to the appropriate attributes here.
    """

    def __init__(
        self,
        prompt: str | list[str] | None = None,
        neg_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        steps: int | None = None,
        guidance_scale: float | list[float] | sfnp.device_array | None = None,
        seed: int | list[int] | None = None,
        input_ids: list[list[int]] | list[list[list[int]]] | list[sfnp.device_array] | None = None,
        sample: sfnp.device_array | None = None,
        prompt_embeds: sfnp.device_array | None = None,
        text_embeds: sfnp.device_array | None = None,
        timesteps: sfnp.device_array | None = None,
        time_ids: sfnp.device_array | None = None,
        denoised_latents: sfnp.device_array | None = None,
        image_array: sfnp.device_array | None = None,
    ):
        super().__init__()
        self.command_buffer = None
        self.print_debug = True
        self.batch_size = 1
        self.phases = {}
        self.phase = None
        self.height = height
        self.width = width

        # Phase inputs:
        # Prep phase.
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.height = height
        self.width = width
        self.seed = seed

        # Encode phase.
        # This is a list of sequenced positive and negative token ids and pooler token ids (tokenizer outputs)
        self.input_ids = input_ids

        # Denoise phase.
        self.prompt_embeds = prompt_embeds
        self.text_embeds = text_embeds
        self.sample = sample
        self.steps = steps
        self.steps_arr = None
        self.timesteps = timesteps
        self.time_ids = time_ids
        self.guidance_scale = guidance_scale

        # Decode phase.
        self.denoised_latents = denoised_latents

        # Postprocess.
        self.image_array = image_array

        self.result_image = None
        self.img_metadata = None

        self.done = sf.VoidFuture()

        # Response control.
        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = True

        self.post_init()

    def set_command_buffer(self, cb):
        if self.input_ids:
            # Take a batch of sets of input ids as ndarrays and fill cb.input_ids
            host_arrs = [None] * 4
            for idx, arr in enumerate(cb.input_ids):
                host_arrs[idx] = arr.for_transfer()
                for i in range(cb.sample.shape[0]):
                    with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                        # TODO: fix this attr redundancy
                        np_arr = self.input_ids[i][idx]

                        m.fill(np_arr)
                cb.input_ids[idx].copy_from(host_arrs[idx])
        steps_arr = list(range(0, self.steps))
        steps_host = cb.steps_arr.for_transfer()
        steps_host.items = steps_arr
        cb.steps_arr.copy_from(steps_host)
        guidance_host = cb.guidance_scale.for_transfer()
        with guidance_host.map(discard=True) as m:
            # TODO: do this without numpy
            np_arr = np.asarray(self.guidance_scale, dtype="float16")

            m.fill(np_arr)
        cb.guidance_scale.copy_from(guidance_host)
        if self.sample:
            sample_host = cb.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(self.sample.tobytes())
            cb.sample.copy_from(sample_host)
        self.command_buffer = cb
        return
        
    def post_init(self):
        """Determines necessary inference phases and tags them with static program parameters."""
        for p in reversed(list(InferencePhase)):
            required, metadata = self.check_phase(p)
            p_data = {"required": required, "metadata": metadata}
            self.phases[p] = p_data
            if not required:
                if p not in [
                    InferencePhase.ENCODE,
                    InferencePhase.PREPARE,
                ]:
                    break
            self.phase = p

    def check_phase(self, phase: InferencePhase):
        match phase:
            case InferencePhase.POSTPROCESS:
                return True, None
            case InferencePhase.DECODE:
                required = not self.image_array
                meta = [self.width, self.height]
                return required, meta
            case InferencePhase.DENOISE:
                required = not self.denoised_latents
                meta = [self.width, self.height, self.steps]
                return required, meta
            case InferencePhase.ENCODE:
                p_results = [
                    self.prompt_embeds,
                    self.text_embeds,
                ]
                required = any([inp is None for inp in p_results])
                return required, None
            case InferencePhase.PREPARE:
                p_results = [self.sample, self.input_ids]
                required = any([inp is None for inp in p_results])
                return required, None

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = None
        self.phases = None
        self.done = sf.VoidFuture()
        self.return_host_array = True

    @staticmethod
    def from_batch(gen_req: GenerateReqInput, index: int) -> "InferenceExecRequest":
        gen_inputs = [
            "prompt",
            "neg_prompt",
            "height",
            "width",
            "steps",
            "guidance_scale",
            "seed",
            "input_ids",
        ]
        rec_inputs = {}
        for item in gen_inputs:
            received = getattr(gen_req, item, None)
            if isinstance(received, list):
                if index >= (len(received)):
                    if len(received) == 1:
                        rec_input = received[0]
                    else:
                        logging.error(
                            "Inputs in request must be singular or as many as the list of prompts."
                        )
                else:
                    rec_input = received[index]
            else:
                rec_input = received
            rec_inputs[item] = rec_input
        req = InferenceExecRequest(**rec_inputs)
        return req


class StrobeMessage(sf.Message):
    """Sent to strobe a queue with fake activity (generate a wakeup)."""

    ...
