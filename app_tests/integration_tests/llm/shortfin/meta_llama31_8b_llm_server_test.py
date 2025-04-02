"""Main test module for LLM server functionality."""

import json
import logging
import pytest
import requests
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from shortfin_apps.llm.components.io_struct import (
    PromptResponse,
    GeneratedResponse,
    GenerateReqOutput,
)
from typing import Any

logger = logging.getLogger(__name__)

from ..model_management import AccuracyValidationException, ModelConfig


pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        (ModelConfig.get(name="llama3.1_8b"), {"prefix_sharing": "none"}),
        (ModelConfig.get(name="llama3.1_8b"), {"prefix_sharing": "trie"}),
    ],
    ids=[
        "llama31_8b_none",
        "llama31_8b_trie",
    ],
    indirect=True,
)


class TestLLMServer:
    """Test suite for LLM server functionality."""

    def test_basic_generation(self, server: tuple[Any, int]) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        expected_prefix = "6 7 8"
        response = self._generate("1 2 3 4 5 ", port)
        response = json.loads(response)
        response = GenerateReqOutput(**response)
        response = PromptResponse(**response.responses[0])
        response = GeneratedResponse(**response.responses[0])
        response = response.text
        if not response.startswith(expected_prefix):
            raise AccuracyValidationException(
                expected=f"{expected_prefix}...",
                actual=response,
                message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
            )

    @pytest.mark.parametrize("encoded_prompt", ["0 1 2 3 4 5 "], indirect=True)
    def test_basic_generation_input_ids(
        self, server: tuple[Any, int], encoded_prompt
    ) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        expected_prefix = "6 7 8"
        response = self._generate(encoded_prompt, port, input_ids=True)
        response = json.loads(response)
        response = GenerateReqOutput(**response)
        response = PromptResponse(**response.responses[0])
        response = GeneratedResponse(**response.responses[0])
        response = response.text
        if not response.startswith(expected_prefix):
            raise AccuracyValidationException(
                expected=f"{expected_prefix}...",
                actual=response,
                message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
            )

    @pytest.mark.parametrize(
        "concurrent_requests",
        [
            2,
            4,
        ],
    )
    def test_concurrent_generation(
        self, server: tuple[Any, int], concurrent_requests: int
    ) -> None:
        """Tests concurrent text generation requests.

        Args:
            server: Tuple of (process, port) from server fixture
            concurrent_requests: Number of concurrent requests to test
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        prompt = "1 2 3 4 5 "
        expected_prefix = "6 7 8"

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self._generate, prompt, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                response = json.loads(response)
                response = GenerateReqOutput(**response)
                response = PromptResponse(**response.responses[0])
                response = GeneratedResponse(**response.responses[0])
                response = response.text
                if not response.startswith(expected_prefix):
                    raise AccuracyValidationException(
                        expected=f"{expected_prefix}...",
                        actual=response,
                        message=f"Concurrent generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
                    )

    def _generate(self, prompt: str | list[int], port: int, input_ids=False) -> str:
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
