# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import Namespace
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SGLangBenchmarkArgs:
    base_url: str
    num_prompt: int
    request_rate: int
    tokenizer: str | Path

    seed: int = 1
    extra_request_body: str | None = None
    output_file: str | Path | None = None
    port: int = 8000
    backend: str = "shortfin"

    def as_namespace(self) -> Namespace:
        return Namespace(
            num_prompts=self.num_prompt,
            base_url=self.base_url,
            tokenizer=str(self.tokenizer),
            request_rate=self.request_rate,
            backend=self.backend,
            output_file=self.output_file,
            seed=self.seed,
            extra_request_body=self.extra_request_body,
            port=8000,
            model=None,
            dataset_name="sharegpt",
            random_input_len=None,
            random_output_len=None,
            random_range_ratio=0.0,
            dataset_path="",
            sharegpt_output_len=None,
            multi=False,
            disable_tqdm=False,
            disable_stream=False,
            disable_ignore_eos=False,
            lora_name=None,
            profile=False,
            sharegpt_context_len=None,
            apply_chat_template=False,
            return_logprob=False,
        )

    def __repr__(self):
        return (
            f"Backend: {self.backend}\n"
            f"Base URL: {self.base_url}\n"
            f"Num Prompt: {self.num_prompt}\n"
            f"Tokenizer: {self.tokenizer}\n"
            f"Request Rate: {self.request_rate}"
        )


def log_jsonl_result(file_path):
    with open(file_path, "r") as file:
        json_string = file.readline().strip()

    json_data = json.loads(json_string)
    for key, val in json_data.items():
        logger.info(f"{key.upper()}: {val}")
