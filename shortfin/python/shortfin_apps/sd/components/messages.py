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
        input_ids: list[list[int]]
        | list[list[list[int]]]
        | list[sfnp.device_array]
        | None = None,
        sample: sfnp.device_array | None = None,
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
        self.sample = sample
        self.guidance_scale = guidance_scale
        self.steps = steps

        # Decode phase.
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
        # Input IDs for CLIP if they are used as inputs instead of prompts.
        if self.input_ids is not None:
            # Take a batch of sets of input ids as ndarrays and fill cb.input_ids
            host_arrs = [None] * len(cb.input_ids)
            for idx, arr in enumerate(cb.input_ids):
                host_arrs[idx] = arr.for_transfer()
                with host_arrs[idx].map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = self.input_ids[0][idx]

                    m.fill(np_arr)
                cb.input_ids[idx].copy_from(host_arrs[idx])

        # Same for noisy latents if they are explicitly provided as a numpy array.
        if isinstance(self.sample, list):
            sample_host = cb.sample.for_transfer()
            for idx, i in enumerate(self.sample):
                with sample_host.view(idx).map(discard=True) as m:
                    m.fill(i.tobytes())
            cb.sample.copy_from(sample_host)
        elif self.sample is not None:
            sample_host = cb.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(self.sample.tobytes())
            cb.sample.copy_from(sample_host)

        # Copy other inference parameters for denoise to device arrays.
        steps_arr = list(range(0, self.steps))
        steps_host = cb.steps_arr.for_transfer()
        steps_host.items = steps_arr
        cb.steps_arr.copy_from(steps_host)

        num_step_host = cb.num_steps.for_transfer()
        num_step_host.items = [self.steps]
        cb.num_steps.copy_from(num_step_host)

        guidance_host = cb.guidance_scale.for_transfer()
        with guidance_host.map(discard=True) as m:
            # TODO: do this without numpy
            np_arr = np.asarray(self.guidance_scale, dtype="float16")

            m.fill(np_arr)
        cb.guidance_scale.copy_from(guidance_host)

        self.command_buffer = cb
        return

    def post_init(self):
        """Determines necessary inference phases and tags them with static program parameters."""
        if self.prompt is not None:
            self.batch_size = len(self.prompt) if isinstance(self.prompt, list) else 1
        elif self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size = len(self.input_ids)
            else:
                self.batch_size = 1
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
                required = True
                meta = [self.width, self.height, self.steps]
                return required, meta
            case InferencePhase.ENCODE:
                required = True
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
