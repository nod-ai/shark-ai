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


parameterization = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        (ModelConfig.get(name="open_llama_3b"), {"prefix_sharing": "none"}),
        (ModelConfig.get(name="open_llama_3b"), {"prefix_sharing": "trie"}),
    ],
    ids=[
        "open_llama_3b_none",
        "open_llama_3b_trie",
    ],
    indirect=True,
)

# Failure:
# > error: Vector shape: [1, 1, 1, 100] does not match the layout (nested_layout<subgroup_tile = [1, 1, 1, 1], batch_tile = [1, 1, 1, 3], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 1, 4], element_tile = [1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0], thread_strides = [0, 0, 0, 16]>) at dim 3. Dimension expected by layout: 96 actual: 100
# > ...
# > torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%397, %398, %399, %float0.000000e00, %false_224, %266, %none_225) : (!torch.vtensor<[1,32,?,100],f16>, !torch.vtensor<[1,32,?,100],f16>, !torch.vtensor<[1,32,?,100],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,32,?,100],f16>, !torch.vtensor<[1,32,?],f32>)
# > error: failed to run translation of source executable to target executable for backend #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
ireebump_xfail = pytest.mark.xfail(
    reason="Compile failure tracked at https://github.com/iree-org/iree/issues/20365",
)

pytestmark = [parameterization, ireebump_xfail]


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
        if not response.text.startswith(expected_prefix):
            raise AccuracyValidationException(
                expected=f"{expected_prefix}...",
                actual=response,
                message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
            )

    @pytest.mark.parametrize("concurrent_requests", [2, 4])
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
