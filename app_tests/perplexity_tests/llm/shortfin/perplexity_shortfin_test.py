import pytest
import numpy as np
import asyncio
import shortfin as sf
import uuid
import requests

from app_tests.integration_tests.llm.server_management import (
    ServerInstance,
    ServerConfig,
)
from app_tests.integration_tests.llm.model_management import TEST_MODELS, ModelProcessor
from app_tests.integration_tests.llm.device_settings import CPU
from shortfin_apps.llm.components.messages import (
    InferencePhase,
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components import generate


pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        ["llama3.1_8b", {"prefix_sharing": "none"}],
    ],
    indirect=True,
)


# class BatchConsistencyTestProcess(sf.Process):
#     """Process to test consistency of results across different batch sizes.
#     This is necessary because InferenceExecRequest uses shortfin.VoidFuture
#     which can only be created on a process (which belongs to a fiber that a worker works through).
#     """

#     def __init__(self, service, input_tokens, batch_sizes, max_response_length):
#         super().__init__(fiber=service.main_fiber)
#         self.service = service
#         self.input_tokens = input_tokens
#         self.batch_sizes = batch_sizes
#         self.max_response_length = max_response_length
#         self.results = {}  # Store results for each batch size
#         # TODO: modify the batcher to guarantee the batch we send isn't split by strobe messages

#     async def run(self):
#         print('batch_sizes:', self.batch_sizes)
#         for batch_size in self.batch_sizes:
#             batch_results = []
#             for _ in range(batch_size):
#                 prefill_req = LlmInferenceExecRequest(
#                     phase=InferencePhase.PREFILL,
#                     input_token_ids=self.input_tokens,
#                     rid=f"test-{batch_size}",
#                 )
#                 prefill_req.return_host_array = True
#                 self.service.batcher.submit(prefill_req)
#                 await prefill_req.done
#                 result_sequence = [prefill_req.result_logits.items]
#                 # first_token = np.argmax(prefill_req.result_logits.items)
#                 # result_sequence = [first_token]

#                 # decode_req = prefill_req
#                 # for _ in range(self.max_response_length - 1):
#                 #     decode_req.reset(InferencePhase.DECODE)
#                 #     decode_req.input_token_ids.append(first_token)
#                 #     decode_req.start_position += 1
#                 #     self.service.batcher.submit(decode_req)
#                 #     await decode_req.done
#                 #     next_token = np.argmax(decode_req.result_logits.items)
#                 #     result_sequence.append(next_token)
#                 #     first_token = next_token
#                 batch_results.append(result_sequence)
#                 # batch_results.append(result_sequence)
#                 # decode_req.free_cache_pages()

#             self.results[batch_size] = batch_results

#         #     first_result = batch_results[0]
#         #     for result in batch_results[1:]:
#         #         assert np.array_equal(
#         #             first_result, result
#         #         ), f"Inconsistent results within batch size {batch_size}"

#         first_batch_result = self.results[self.batch_sizes[0]][0]
#         logger.info("TESTING")
#         # first_batch_result = self.results[self.batch_sizes[0]][0]
#         # for batch_size in self.batch_sizes[1:]:
#         #     assert np.array_equal(
#         #         first_batch_result, self.results[batch_size][0]
#         #     ), f"Inconsistent results between batch sizes {self.batch_sizes[0]} and {batch_size}"


# def test_batch_and_nobatch_consistency(model_artifacts, server):
#     """
#     Test that requests produce identical results regardless of batch size.
#     If this test fails, it means that changing the batch size changes the generation results.
#     Look for kvcache corruption due to
#     - improper seq_len / current_position handling in service.py
#     - improper masking in sharktank
#     """
#     # Create and run the test process
#     test_process = BatchConsistencyTestProcess(
#         server,
#         input_tokens=[1, 2, 3, 4],
#         # batch_sizes=[1, 2, 3, 4],
#         batch_sizes=[1],
#         max_response_length=3,
#     )
#     test_process.launch()


# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import json
import logging

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin_apps.llm.components.io_struct import GenerateReqInput
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.service import GenerateService
from shortfin_apps.llm.components.tokenizer import Encoding

logger = logging.getLogger(__name__)


def generate_request(prompt: str | list[int], port: int, input_ids=False) -> str:
    """Helper method to make generation request to server.

    Args:
        prompt: Input text prompt
        port: Server port number

    Returns:
        Generated text response

    Raises:
        requests.exceptions.RequestException: If request fails
        AccuracyValidationException: If response format is invalid
    """
    logits = None
    payload = {
        "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
        "rid": uuid.uuid4().hex,
        "stream": False,
        "return_logprob": True,  # dummy param in /home/aramalin/shark-ai/shortfin/python/shortfin_apps/llm/components/io_struct.py needs to be plumbed thro' to return logits
    }
    generate_url = f"http://localhost:{port}/generate"
    if input_ids:
        payload["input_ids"] = prompt
    else:
        payload["text"] = prompt
    try:
        response = requests.post(
            generate_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            # timeout=30,  # Add reasonable timeout
        )
        response.raise_for_status()

        # TODO: Parse response for logits
        logits = response.text
        print(logits)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return logits


def test_prefill_logits_from_server(model_artifacts, server):
    """
    Test that requests produce identical results regardless of batch size.
    If this test fails, it means that changing the batch size changes the generation results.
    Look for kvcache corruption due to
    - improper seq_len / current_position handling in service.py
    - improper masking in sharktank
    """
    # Create and run the test process
    # test_process = GenerateItemProcess(
    #     server,
    #     input_tokens=[1, 2, 3, 4],
    #     # batch_sizes=[1, 2, 3, 4],
    #     batch_sizes=[1],
    #     max_response_length=3,
    # )
    process, port = server
    print("PORT:", port)
    gen_req = generate_request(prompt="1, 2, 3, 4, 5", port=port)
    gen_req.post_init()
    responder = FastAPIResponder(request)
    client_process = generate.ClientGenerateBatchProcess(server, gen_req, responder)
    client_process.launch()
