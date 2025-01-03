"""Main test module for LLM server functionality."""
import pytest
import requests
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TestLLMServer:
    """Test suite for LLM server functionality."""

    @pytest.mark.parametrize(
        "model_artifacts,server",
        [
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "none"}),
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "trie"}),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "none"},
                marks=pytest.mark.xfail(
                    reason="llama3.1_8b irpa file not available on CI machine"
                ),
            ),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "trie"},
                marks=pytest.mark.xfail(
                    reason="llama3.1_8b irpa file not available on CI machine"
                ),
            ),
        ],
        ids=[
            "open_llama_3b_none",
            "open_llama_3b_trie",
            "llama31_8b_none",
            "llama31_8b_trie",
        ],
        indirect=True,
    )
    def test_basic_generation(self, server: tuple[Any, int]) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        response = self._generate("1 2 3 4 5 ", port)
        assert response.startswith("6 7 8"), f"Unexpected response: {response}"

    @pytest.mark.parametrize(
        "model_artifacts,server",
        [
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "none"}),
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "trie"}),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("concurrent_requests", [2, 4, 8])
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
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self._generate, prompt, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                assert response.startswith("6 7 8"), f"Unexpected response: {response}"

    def _generate(self, prompt: str, port: int) -> str:
        """Helper method to make generation request to server.

        Args:
            prompt: Input text prompt
            port: Server port number

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json={
                "text": prompt,
                "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
                "rid": uuid.uuid4().hex,
                "stream": False,
            },
            timeout=30,  # Add reasonable timeout
        )
        response.raise_for_status()

        # Parse streaming response
        data = response.text
        assert data.startswith("data: "), f"Invalid response format: {data}"
        return data[6:].rstrip("\n")
