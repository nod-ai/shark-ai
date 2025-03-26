"""
Test for sharded Meta LLaMa 3.1 8B model with tensor parallelism.

This test validates tensor parallelism with a model large enough to exercise
sharding. Note: This test uses LLaMa 3.1 8B as a workaround for
the TinyStories 25M model failing with  `Memory access fault by GPU node-3 (Agent handle: 0x555c24e83f80) on address 0x7fc28a1e4000. Reason: Unknown`

The test should run on either CPU or GPU.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging
import os
import time
import pytest
import requests
import uuid

logger = logging.getLogger(__name__)

from ..model_management import AccuracyValidationException
from ..server_management import ServerInstance

MODEL_NAME = "llama3.1_8b_tp2"


@pytest.fixture(scope="function")
def device_type(test_device):
    """Determine if we're running on CPU or GPU based on test_device"""
    return "gpu" if "gfx" in test_device else "cpu"


pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        [MODEL_NAME, {"prefix_sharing": "none"}],
    ],
    indirect=True,
)

# Test prompts appropriate for LLaMa 3.1
PROMPT = "What is the capital of the United States?"
EXPECTED_PATTERN = (
    "Washington"  # just a sanity check. pass the model if it mentions Washington at all
)


class TestShardedLlama31Server:
    """Tests sharded model server functionality on both CPU and GPU devices."""

    def test_sharded(self, server: ServerInstance, test_device, device_type) -> None:
        """Tests single request generation with a sharded model.

        Validates basic sharding functionality with a single request.

        Args:
            server: ServerInstance from server fixture
            test_device: The device being tested (from pytest fixture)
            device_type: Simplified device type (cpu/gpu) from fixture
        """
        process, port = server.process, server.port
        assert process.poll() is None, "Server process terminated unexpectedly"

        logger.info(
            f"Testing with {test_device} (type: {device_type})" f"single request"
        )

        try:
            # Send a single request
            response = self._generate(PROMPT, port)
            if EXPECTED_PATTERN not in response:
                raise AccuracyValidationException(
                    expected=f"Response containing '{EXPECTED_PATTERN}'",
                    actual=response,
                    message=f"Generation did not contain expected pattern.\nExpected to contain: {EXPECTED_PATTERN}\nActual response: {response}",
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # On connection errors, dump server logs
            logger.error("Connection error or timeout occurred: %s", str(e))
            server._dump_log_tail(lines=50)
            raise

    def test_concurrent_generation_sharded(
        self, server: ServerInstance, test_device, device_type
    ) -> None:
        """Tests concurrent text generation with a sharded model.

        Validates that the server can handle multiple simultaneous requests
        with tensor parallelism enabled.

        Args:
            server: Tuple of (process, port, server_instance) from server fixture
            test_device: The device being tested (from pytest fixture)
            device_type: Simplified device type (cpu/gpu) from fixture
        """
        process, port = server.process, server.port
        assert process.poll() is None, "Server process terminated unexpectedly"

        concurrent_requests = 3

        logger.info(
            f"Testing with {test_device} (type: {device_type})"
            f"concurrent_requests: {concurrent_requests}"
        )

        try:
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [
                    executor.submit(self._generate, PROMPT, port)
                    for _ in range(concurrent_requests)
                ]

                for future in as_completed(futures):
                    try:
                        response = future.result()
                        if EXPECTED_PATTERN not in response:
                            raise AccuracyValidationException(
                                expected=f"Response containing '{EXPECTED_PATTERN}'",
                                actual=response,
                                message=f"Generation did not contain expected pattern.\nExpected to contain: {EXPECTED_PATTERN}\nActual response: {response}",
                            )
                    except (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                    ) as e:
                        # On connection errors, dump server logs
                        logger.error("Connection error or timeout occurred: %s", str(e))
                        server._dump_log_tail(lines=50)
                        raise
        except Exception as e:
            # Catch any other errors and dump logs
            logger.error("Error during concurrent generation: %s", str(e))
            server._dump_log_tail(lines=50)
            raise

    def _generate(self, prompt: str, port: int) -> str:
        """Make generation request to server.

        Args:
            prompt: Input text prompt
            port: Server port number

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If request fails
            AccuracyValidationException: If response format is invalid
        """
        time.sleep(15)
        request_id = uuid.uuid4().hex
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_completion_tokens": 15,
                "temperature": 0.0,
            },
            "rid": request_id,
            "stream": False,
        }

        logger.info("Sending request with ID: %s to port %d", request_id, port)

        try:
            response = requests.post(
                f"http://localhost:{port}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,  # 30 second timeout
            )
            response.raise_for_status()

            data = response.text
            if not data.startswith("data: "):
                raise AccuracyValidationException(
                    expected="Response starting with 'data: '",
                    actual=data,
                    message=f"Invalid response format.\nExpected format starting with 'data: '\nActual response: {data}",
                )

            logger.info("Received successful response for request ID: %s", request_id)
            return data[6:].rstrip("\n")

        except requests.exceptions.Timeout:
            logger.error("Request timed out for ID: %s", request_id)
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error for request ID: %s - %s", request_id, str(e))
            raise
