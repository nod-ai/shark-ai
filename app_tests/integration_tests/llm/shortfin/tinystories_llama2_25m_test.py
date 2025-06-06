"""
Simple smoke tests to:
- ensure the full fastapi server works
- ensure the smoke test model works so we know it's not a model issue when another test using this model fails.
"""

import json
import logging
import pytest
import requests
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from shortfin_apps.llm.components.io_struct import (
    PromptResponse,
    GeneratedResponse,
    GenerateReqOutput,
)
import urllib3

logger = logging.getLogger(__name__)

from ..model_management import AccuracyValidationException, ModelConfig


pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        (ModelConfig.get(name="tinystories_llama2_25m"), {"prefix_sharing": "none"}),
        (
            ModelConfig.get(name="tinystories_llama2_25m"),
            {
                "token_selection_strategy": "independent",
                "prefix_sharing": "none",
                "num_beams": 2,
            },
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m"),
            {
                "prefix_sharing": "none",
                "token_selection_strategy": "beam_search",
                "num_beams": 2,
            },
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_argmax"),
            {"prefix_sharing": "none"},
        ),
        (
            ModelConfig.get(name="tinystories_llama2_25m_gpu_topk_k4"),
            {"prefix_sharing": "none"},
        ),
    ],
    ids=[
        "tinystories_llama2_25m_none",
        "tinystories_llama2_25m_none_independent_2_beams",
        "tinystories_llama2_25m_none_beam_search_2_beams",
        "tinystories_llama2_25m_gpu_argmax_none",
        "tinystories_llama2_25m_gpu_topk_k4_none",
    ],
    indirect=True,
)


# goldens are generated in: https://colab.research.google.com/drive/1pFiyvyIxk1RsHnw5gTk_gu9QiQNy9gfW?usp=sharing
GOLDEN_PROMPT = "Once upon a time"
GOLDEN_RESPONSE = {
    ", there was a little girl named Lily. She loved to play with her"
}  # this assumes purely deterministic greedy search

GOLDEN_BEAM_SEARCH_RESPONSE = {
    ", there was a little girl named Lily. She loved to play with her",
    ", there was a little girl named Lily. She had a big, red",
}  # this assumes purely deterministic beam search with 2 beams


class TestLLMServer:
    """Test suite for LLM server functionality."""

    def test_basic_generation(
        self, request: pytest.FixtureRequest, server: tuple[Any, int]
    ) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        test_id = request.node.callspec.id

        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly"
        prompt = GOLDEN_PROMPT
        expected_response = (
            GOLDEN_RESPONSE
            if config.token_selection_strategy != "beam_search"
            else GOLDEN_BEAM_SEARCH_RESPONSE
        )

        try:
            response = self._generate(prompt, port)
        except Exception as e:
            if "gpu_topk_k4" in test_id:
                pytest.xfail(
                    "(https://github.com/iree-org/iree/issues/20772): Current top-k kernel is slow and causes `ReadTimeout`"
                )
            raise e

        response = json.loads(response)
        req_output = GenerateReqOutput(**response)

        for prompt_response in req_output.responses:
            prompt_response = PromptResponse(**prompt_response)
            assert len(prompt_response.responses) == config.num_beams
            for generated_response in prompt_response.responses:
                generated_response = GeneratedResponse(**generated_response)
                response_text = generated_response.text
                if response_text not in expected_response:
                    raise AccuracyValidationException(
                        expected=f"{expected_response}...",
                        actual=response_text,
                        message=f"Generation did not match expected pattern.\nExpected to be one of: {expected_response}\nActual response: {response_text}",
                    )

    @pytest.mark.parametrize(
        "concurrent_requests",
        [
            2,
            4,
        ],
    )
    def test_concurrent_generation(
        self,
        request: pytest.FixtureRequest,
        server: tuple[Any, int],
        concurrent_requests: int,
    ) -> None:
        """Tests concurrent text generation requests.

        Args:
            server: Tuple of (process, port) from server fixture
            concurrent_requests: Number of concurrent requests to test
        """
        test_id = request.node.callspec.id

        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        prompt = GOLDEN_PROMPT
        expected_response = (
            GOLDEN_RESPONSE
            if config.token_selection_strategy != "beam_search"
            else GOLDEN_BEAM_SEARCH_RESPONSE
        )

        def _generate_task(prompt: str, port: int):
            try:
                return self._generate(prompt, port)
            except Exception as e:
                if "gpu_topk_k4" in test_id:
                    pytest.xfail(
                        "(https://github.com/iree-org/iree/issues/20772): Current top-k kernel is slow and causes `ReadTimeout`"
                    )

                raise e

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(_generate_task, prompt, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                response = json.loads(response)
                req_output = GenerateReqOutput(**response)

                for prompt_response in req_output.responses:
                    prompt_response = PromptResponse(**prompt_response)
                    assert len(prompt_response.responses) == config.num_beams

                    for generated_response in prompt_response.responses:
                        generated_response = GeneratedResponse(**generated_response)
                        generated_text = generated_response.text
                        if generated_text not in expected_response:
                            raise AccuracyValidationException(
                                expected=f"{expected_response}...",
                                actual=response,
                                message=f"Concurrent generation did not match expected pattern.\nExpected to start with: {expected_response}\nActual response: {response}",
                            )

    def _generate(
        self,
        prompt: str | list[int],
        port: int,
        input_ids: bool = False,
    ) -> str:
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
        payload = {
            "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
            "rid": uuid.uuid4().hex,
            "stream": False,
        }
        if input_ids:
            payload["input_ids"] = prompt
        else:
            payload["text"] = prompt
        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,  # Add reasonable timeout
        )
        response.raise_for_status()
        return response.text
