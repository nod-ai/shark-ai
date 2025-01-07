# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import concurrent.futures
import logging
import os
import pytest
import requests
import uuid

from ..utils import AccuracyValidationException, start_log_group, end_log_group

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


def do_generate(prompt, port, concurrent_requests=1):
    logger.info("Generating request...")
    headers = {"Content-Type": "application/json"}
    # Create a GenerateReqInput-like structure
    data = {
        "text": prompt,
        "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
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

    response_data = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrent_requests
    ) as executor:
        futures = [
            executor.submit(
                lambda: requests.post(
                    f"{BASE_URL}/generate", headers=headers, json=data
                )
            )
            for _ in range(concurrent_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            response = future.result()

            logger.info(f"Generate endpoint status code: {response.status_code}")
            if response.status_code == 200:
                logger.info("Generated text:")
                data = response.text
                assert data.startswith("data: ")
                data = data[6:]
                assert data.endswith("\n\n")
                data = data[:-2]
                logger.info(data)
                response_data.append(data)
            else:
                response.raise_for_status()

    return response_data


@pytest.mark.parametrize(
    "model_test_dir,write_config,llm_server",
    [
        pytest.param(
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "none"},
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        ),
        pytest.param(
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "trie"},
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        ),
    ],
    indirect=True,
)
def test_llm_server(llm_server):
    # Here you would typically make requests to your server
    # and assert on the responses
    server, port = llm_server
    assert server.poll() is None
    PROMPT = "1 2 3 4 5 "
    expected_output_prefix = "6 7 8"
    logger.info(
        "Sending HTTP Generation Request"
        + start_log_group("Sending HTTP Generation Request")
    )
    output = do_generate(PROMPT, port)[0]
    # log to GITHUB_STEP_SUMMARY if we are in a GitHub Action
    if "GITHUB_ACTION" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            # log prompt
            f.write("LLM results:\n")
            f.write(f"- llm_prompt:`{PROMPT}`\n")
            f.write(f"- llm_output:`{output}`\n")
    if not output.startswith(expected_output_prefix):
        raise AccuracyValidationException(
            f"Expected '{output}' to start with '{expected_output_prefix}'"
        )
    logger.info("HTTP Generation Request Successful" + end_log_group())


@pytest.mark.parametrize(
    "model_test_dir,write_config,llm_server",
    [
        pytest.param(
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "none"},
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        ),
        pytest.param(
            {
                "repo_id": "SlyEcho/open_llama_3b_v2_gguf",
                "model_file": "open-llama-3b-v2-f16.gguf",
                "tokenizer_id": "openlm-research/open_llama_3b_v2",
                "settings": CPU_SETTINGS,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "trie"},
            {"model_file": "open-llama-3b-v2-f16.gguf", "settings": CPU_SETTINGS},
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "concurrent_requests",
    [2, 4, 8],
)
@pytest.mark.xfail(
    raises=AccuracyValidationException,
    reason="Concurreny issues in Shortfin batch processing",
)
def test_llm_server_concurrent(llm_server, concurrent_requests):
    logger.info("Testing concurrent invocations")
    server, port = llm_server
    assert server.poll() is None
    PROMPT = "1 2 3 4 5 "
    expected_output_prefix = "6 7 8"
    logger.info(
        "Sending HTTP Generation Request"
        + start_log_group("Sending HTTP Generation Request")
    )
    outputs = do_generate(PROMPT, port, concurrent_requests)

    for output in outputs:
        # log to GITHUB_STEP_SUMMARY if we are in a GitHub Action
        if "GITHUB_ACTION" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                # log prompt
                f.write("LLM results:\n")
                f.write(f"- llm_prompt:`{PROMPT}`\n")
                f.write(f"- llm_output:`{output}`\n")

        if not output.startswith(expected_output_prefix):
            raise AccuracyValidationException(
                f"Expected '{output}' to start with '{expected_output_prefix}'"
            )
        logger.info("HTTP Generation Request Successful" + end_log_group())
