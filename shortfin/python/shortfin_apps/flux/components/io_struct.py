# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional, Union
from dataclasses import dataclass
import uuid


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    prompt: Optional[Union[List[str], str]] = None
    # The input negative prompt. It can be a single prompt or a batch of prompts.
    neg_prompt: Optional[Union[List[str], str]] = None
    # Output image dimensions per prompt.
    height: Optional[Union[List[int], int]] = None
    width: Optional[Union[List[int], int]] = None
    # The number of inference steps; one int per prompt.
    steps: Optional[Union[List[int], int]] = None
    # The classifier-free-guidance scale for denoising; one float per prompt.
    guidance_scale: Optional[Union[List[float], float]] = None
    # The seed for random latents generation; one int per prompt.
    seed: Optional[Union[List[int], int]] = None
    # Token ids: only used in place of prompt.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Negative token ids: only used in place of negative prompt.
    neg_input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Output image format. Defaults to base64. One string ("PIL", "base64")
    output_type: Optional[List[str]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None

    def post_init(self):
        if (self.prompt is None and self.input_ids is None) or (
            self.prompt is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        if isinstance(self.prompt, str):
            self.prompt = [str]

        self.num_output_images = (
            len(self.prompt) if self.prompt is not None else len(self.input_ids)
        )

        batchable_args = [
            self.prompt,
            self.neg_prompt,
            self.height,
            self.width,
            self.steps,
            self.guidance_scale,
            self.seed,
            self.input_ids,
            self.neg_input_ids,
        ]
        for arg in batchable_args:
            if isinstance(arg, list):
                if len(arg) != self.num_output_images and len(arg) != 1:
                    raise ValueError(
                        f"Batchable arguments should either be singular or as many as the full batch ({self.num_output_images})."
                    )
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(self.num_output_images)]
        else:
            if not isinstance(self.rid, list):
                raise ValueError("The rid should be a list.")
        if self.output_type is None:
            self.output_type = ["base64"] * self.num_output_images
        # Temporary restrictions
        heights = [self.height] if not isinstance(self.height, list) else self.height
        widths = [self.width] if not isinstance(self.width, list) else self.width
        if any(dim != 1024 for dim in [*heights, *widths]):
            raise ValueError(
                "Currently, only 1024x1024 output image size is supported."
            )
