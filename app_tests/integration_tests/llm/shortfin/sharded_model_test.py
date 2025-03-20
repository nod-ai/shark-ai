"""
Test for sharded model serving with concurrent requests.
Tests a tensor parallel sharded model with concurrent requests.

The tensor parallelism size is parameterized and can be configured by
changing the TENSOR_PARALLELISM_SIZE constant at the top of this file.
This allows for easily testing with different tensor parallelism sizes (2, 4, 8, etc.)
without having to modify multiple parts of the code.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging
import pytest
import requests
import uuid

logger = logging.getLogger(__name__)

from ..model_management import (
    AccuracyValidationException,
    ModelConfig,
    ModelSource,
    ModelProcessor,
)
from ..device_settings import DeviceSettings

# Define tensor parallelism parameter that can be easily changed in a single place
TENSOR_PARALLELISM_SIZE = 2  # Change this to 2, 4, 8, etc. as needed

# Create a dynamic model name based on the tensor parallelism size
MODEL_NAME = f"tinystories_tp{TENSOR_PARALLELISM_SIZE}"

# Generate device settings for variable tensor parallelism
def create_device_settings(tp_size):
    """Create device settings with proper flags for the specified tensor parallelism size."""
    # Generate device target flags for each device
    compile_flags = [
        "--iree-hip-target=gfx942",
    ]

    # Add one target device flag for each shard
    for i in range(tp_size):
        compile_flags.append(f"--iree-hal-target-device=hip[{i}]")

    # Generate server flags with device ids
    server_flags = ["--device=hip", "--device_ids"]

    # Add device ids (0, 1, 2, ...) for each shard
    for i in range(tp_size):
        server_flags.append(str(i))

    return DeviceSettings(
        compile_flags=tuple(compile_flags),
        server_flags=tuple(server_flags),
    )


# Set up model with the specified tensor parallelism in TEST_MODELS
def setup_model():
    from ..model_management import TEST_MODELS, create_tinystories_model

    # Create model with the specified tensor parallelism size if it doesn't exist
    if MODEL_NAME not in TEST_MODELS:
        TEST_MODELS[MODEL_NAME] = create_tinystories_model(
            tp_size=TENSOR_PARALLELISM_SIZE,
            batch_sizes=(4,),  # Fixed batch size of 4 for testing
        )

    # Set device settings for the model
    TEST_MODELS[MODEL_NAME].device_settings = create_device_settings(
        TENSOR_PARALLELISM_SIZE
    )


# Check for available devices before running the test
import os
import subprocess


def check_available_gpus():
    """Check how many GPU devices are available"""
    try:
        # Try to use rocm-smi to get GPU count - works on ROCm systems
        result = subprocess.run(
            ["rocm-smi", "--showallinfo"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            # Count the occurrences of "GPU" in the output
            gpu_count = result.stdout.count("GPU")
            return gpu_count

        # Fall back to environment variable CUDA_VISIBLE_DEVICES if available
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            return len(devices)

        # Default to assuming we have 4 GPUs if we can't determine
        return 4
    except Exception as e:
        logger.warning(f"Failed to check GPU count: {e}")
        return 4  # Assume 4 GPUs if check fails


# Check if we have enough GPUs for the requested tensor parallelism
available_gpus = check_available_gpus()
if TENSOR_PARALLELISM_SIZE > available_gpus:
    logger.warning(
        f"Requested tensor parallelism size {TENSOR_PARALLELISM_SIZE} exceeds "
        f"available GPU count {available_gpus}. Test may fail if not enough devices available."
    )

# Setup the model configuration
setup_model()

# Parametrize the test to use our dynamically named model
pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        [MODEL_NAME, {"prefix_sharing": "none"}],
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
        multiple requests with a tensor-parallel sharded model.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        concurrent_requests = 3  # Fixed number of concurrent requests

        logger.info(
            f"Testing with tensor parallelism={TENSOR_PARALLELISM_SIZE}, concurrent_requests={concurrent_requests}"
        )

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

        return data[6:].rstrip("\n")
