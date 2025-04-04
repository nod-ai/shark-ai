# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

from typing import Union

import logging

from PIL.Image import Image
import shortfin as sf
import shortfin.array as sfnp

from .io_struct import GenerateReqInput
from ...utils import InferenceExecRequest

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


class FluxInferenceExecRequest(InferenceExecRequest):
    """
    Generalized request passed for an individual phase of image generation.

    Used for individual image requests. Bundled as lists by the batcher for inference processes,
    and inputs joined for programs with bs>1.

    Inference execution processes are responsible for writing their outputs directly to the appropriate attributes here.
    """

    def __init__(
        self,
        prompt: str | None = None,
        neg_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        steps: int | None = None,
        guidance_scale: float | sfnp.device_array | None = None,
        seed: int | None = None,
        cfg_mult: int = 1,
        clip_input_ids: list[list[int]] | None = None,
        t5xxl_input_ids: list[list[int]] | None = None,
        sample: sfnp.device_array | None = None,
        txt: sfnp.device_array | None = None,
        vec: sfnp.device_array | None = None,
        img_ids: sfnp.device_array | None = None,
        txt_ids: sfnp.device_array | None = None,
        timesteps: sfnp.device_array | None = None,
        denoised_latents: sfnp.device_array | None = None,
        image_array: sfnp.device_array | None = None,
    ):
        super().__init__()
        self.print_debug = True

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
        self.cfg_mult = 1

        # Encode phase.
        # This is a list of sequenced positive and negative token ids and pooler token ids.
        self.clip_input_ids = clip_input_ids
        self.t5xxl_input_ids = t5xxl_input_ids
        self.sample = sample

        # Denoise phase.
        self.img = None
        self.txt = txt
        self.vec = vec
        self.img_ids = img_ids
        self.txt_ids = txt_ids
        self.steps = steps
        self.timesteps = timesteps
        self.guidance_scale = guidance_scale

        # Decode phase.
        self.denoised_latents = denoised_latents

        # Postprocess.
        self.image_array = image_array

        self.response_image: Union[Image, None] = None

        self.done = sf.VoidFuture()

        # Response control.
        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = True

        self.post_init()

    @staticmethod
    def from_batch(gen_req: GenerateReqInput, index: int) -> "FluxInferenceExecRequest":
        gen_inputs = [
            "prompt",
            "neg_prompt",
            "height",
            "width",
            "steps",
            "guidance_scale",
            "seed",
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
        return FluxInferenceExecRequest(**rec_inputs)

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
                    self.txt,
                    self.vec,
                ]
                required = any([inp is None for inp in p_results])
                return required, None
            case InferencePhase.PREPARE:
                p_results = [self.sample, self.clip_input_ids, self.t5xxl_input_ids]
                required = any([inp is None for inp in p_results])
                return required, None

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = None
        self.phases = None
        self.done = sf.VoidFuture()
        self.return_host_array = True
