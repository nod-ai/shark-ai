# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import pytest
import requests
import uuid

from .utils import AccuracyValidationException

logger = logging.getLogger(__name__)

CPU_SETTINGS = {
    "device_flags": [
        "-iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ],
    "device": "local-task",
}
IREE_HIP_TARGET = os.environ.get("IREE_HIP_TARGET", "gfx1100")
gpu_settings = {
    "device_flags": [
        "-iree-hal-target-backends=rocm",
        f"--iree-hip-target={IREE_HIP_TARGET}",
    ],
    "device": "hip",
}


def do_generate(prompt, port):
    logger.info("Generating request...")
    headers = {"Content-Type": "application/json"}
    # Create a GenerateReqInput-like structure
    data = {
        "text": prompt,
        "sampling_params": {"max_completion_tokens": 50, "temperature": 0.7},
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    logger.info("Prompt text:")
    logger.info(data["text"])
    BASE_URL = f"http://localhost:{port}"
    response = requests.post(f"{BASE_URL}/generate", headers=headers, json=data)
    logger.info(f"Generate endpoint status code: {response.status_code}")
    if response.status_code == 200:
        logger.info("Generated text:")
        data = response.text
        assert data.startswith("data: ")
        data = data[6:]
        assert data.endswith("\n\n")
        data = data[:-2]
        return data
    else:
        response.raise_for_status()


@pytest.mark.parametrize(
    "model_test_dir,llm_server",
    [
        (
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        )
    ],
    indirect=True,
)
def test_llm_server(llm_server, available_port):
    # Here you would typically make requests to your server
    # and assert on the responses
    assert llm_server.poll() is None
    PROMPT = "1 2 3 4 5 "
    expected_output_prefix = "6 7 8"
    logger.info("::group::Sending HTTP Generation Request")
    output = do_generate(PROMPT, available_port)
    # log to GITHUB_STEP_SUMMARY if we are in a GitHub Action
    if "GITHUB_ACTION" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            # log prompt
            f.write("LLM results:\n")
            f.write(f"- llm_prompt:`{PROMPT}`\n")
            f.write(f"- llm_output:`{output}`\n")
    logger.info(output)
    if not output.startswith(expected_output_prefix):
        raise AccuracyValidationException(
            f"Expected '{output}' to start with '{expected_output_prefix}'"
        )
