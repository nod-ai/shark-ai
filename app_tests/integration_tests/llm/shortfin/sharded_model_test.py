"""
Test for sharded model serving with concurrent requests.
Tests a 4-way sharded model with batch size 4 handling 3 concurrent requests.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging
import pytest
import requests
import uuid

logger = logging.getLogger(__name__)

from ..model_management import AccuracyValidationException, ModelConfig, ModelSource

pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        ["tinystories_tp4", {"prefix_sharing": "none"}],
    ],
    indirect=True,
)

# Test prompt and expected response pattern for tinystories
PROMPT = "Once upon a time, there was a"
EXPECTED_PATTERN = "little"  # Common word in children's stories


class TestShardedModelServer:
    """Test suite for sharded model server functionality."""

    def test_concurrent_generation_sharded(self, server: tuple[Any, int]) -> None:
        """Tests concurrent text generation with a sharded model.

        Uses 3 concurrent requests to test the server's ability to handle
        multiple requests with a 4-way sharded model configured for batch size 4.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        concurrent_requests = 3  # Fixed number of concurrent requests

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self._generate, PROMPT, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                if EXPECTED_PATTERN not in response:
                    raise AccuracyValidationException(
                        expected=f"...{EXPECTED_PATTERN}...",
                        actual=response,
                        message=f"Generation did not contain expected pattern.\nExpected to contain: {EXPECTED_PATTERN}\nActual response: {response}",
                    )

    def _generate(self, prompt: str, port: int) -> str:
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
            "text": prompt,
            "sampling_params": {
                "max_completion_tokens": 15,
                "temperature": 0.0,  # Use greedy sampling for deterministic output
            },
            "rid": uuid.uuid4().hex,
            "stream": False,
        }

        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.text
        if not data.startswith("data: "):
            raise AccuracyValidationException(
                expected="Response starting with 'data: '",
                actual=data,
                message=f"Invalid response format.\nExpected format starting with 'data: '\nActual response: {data}",
            )
        else:
            logger.info("Generation succeeded with expected pattern.")
            logger.info("Prompt: %s", prompt)
            logger.info("Output: %s", data)

        return data[6:].rstrip("\n")
