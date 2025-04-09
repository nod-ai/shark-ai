# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Objects transferred between components.

Portions adapted from API definitions originating in:

sglang: Copyright 2023-2024 SGLang Team, Licensed under the Apache License, Version 2.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import uuid

# TODO: Should max, min, and default change based on the model being ran?
MAX_TEMPERATURE = 2.0
# Source: https://github.com/ggml-org/llama.cpp/blob/master/examples/main/README.md?#temperature
DEFAULT_TEMPERATURE = 0.8
MIN_TEMPERATURE = 0.1


@dataclass
class SamplingParams:
    # Number of parallel samples
    n: int = 1
    # Max tokens to generate during decode loop
    max_completion_tokens: int = 50
    # Temperature to use during generation
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self):
        # Ensure temperature is within acceptable range
        self.temperature = min(MAX_TEMPERATURE, max(self.temperature, MIN_TEMPERATURE))


# Adapted from:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py
@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: List[SamplingParams] | SamplingParams = field(
        default_factory=SamplingParams
    )
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to decode the response before returning it.
    return_input_ids: bool = False
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None

    is_single: bool = True

    def post_init(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")
        if isinstance(self.sampling_params, list) or self.sampling_params.n > 1:
            is_single = False
        else:
            if self.text is not None:
                is_single = isinstance(self.text, str)
            else:
                is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = -1
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
        else:
            parallel_sample_num_list = []
            sampling_params = self.sampling_params
            if isinstance(sampling_params, SamplingParams):
                parallel_sample_num = sampling_params.n
            elif isinstance(sampling_params, list):
                for sp in sampling_params:
                    parallel_sample_num = sp.n
                    parallel_sample_num_list.append(parallel_sample_num)
                parallel_sample_num = max(parallel_sample_num_list)
                all_equal = all(
                    element == parallel_sample_num
                    for element in parallel_sample_num_list
                )
                if parallel_sample_num > 1 and (not all_equal):
                    # TODO cope with the case that the parallel_sample_num is different for different samples
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )
            else:
                parallel_sample_num = 1
            self.parallel_sample_num = parallel_sample_num

            if parallel_sample_num != 1:
                # parallel sampling +1 represents the original prefill stage
                num = parallel_sample_num + 1
                if isinstance(self.text, list):
                    # support batch operation
                    self.batch_size = len(self.text)
                    num = num * len(self.text)
                elif isinstance(self.input_ids, list) and isinstance(
                    self.input_ids[0], list
                ):
                    self.batch_size = len(self.input_ids)
                    num = num * len(self.input_ids)
                else:
                    self.batch_size = 1
            else:
                # support select operation
                num = len(self.text) if self.text is not None else len(self.input_ids)
                self.batch_size = num

            if self.image_data is None:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num

            if not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num


@dataclass
class GeneratedResponse:
    text: str


@dataclass
class PromptResponse:
    prompt: str

    responses: list[GeneratedResponse]


@dataclass
class GenerateReqOutput:
    responses: list[PromptResponse]
