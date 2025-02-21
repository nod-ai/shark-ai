import pytest
import numpy as np
import asyncio
import shortfin as sf

from app_tests.integration_tests.llm.server_management import (
    ServerInstance,
    ServerConfig,
)
from app_tests.integration_tests.llm.model_management import TEST_MODELS, ModelProcessor
from app_tests.integration_tests.llm.device_settings import CPU
from shortfin_apps.llm.components.messages import InferencePhase, InferenceExecRequest


@pytest.fixture
def processor():
    return ModelProcessor(base_dir="/tmp/model_management")


@pytest.fixture
def model_config():
    config = TEST_MODELS["tinystories_llama2_25m"]
    config.device_settings = CPU
    return config


@pytest.fixture
def server_instance(processor, model_config):
    artifacts = processor.process_model(model_config)
    sconf = ServerConfig(
        artifacts=artifacts,
        device_settings=CPU,
        prefix_sharing_algorithm="none",
    )
    sinst = ServerInstance(sconf)
    sinst.port = 0
    return sinst


class BatchConsistencyTestProcess(sf.Process):
    """Process to test consistency of results across different batch sizes."""

    def __init__(self, service, input_tokens, batch_sizes, max_response_length):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.input_tokens = input_tokens
        self.batch_sizes = batch_sizes
        self.max_response_length = max_response_length
        self.results = {}  # Store results for each batch size
        self.service.batcher.strobe_enabled = (
            False  # manually strobe the batcher to launch batches
        )

    async def run(self):
        for batch_size in self.batch_sizes:
            batch_results = []
            for _ in range(batch_size):
                prefill_req = InferenceExecRequest(
                    phase=InferencePhase.PREFILL,
                    input_token_ids=self.input_tokens,
                    rid=f"test-{batch_size}",
                )
                prefill_req.return_host_array = True
                self.service.batcher.submit(prefill_req)
                await prefill_req.done
                first_token = np.argmax(prefill_req.result_logits.items)
                result_sequence = [first_token]

                decode_req = prefill_req
                for _ in range(self.max_response_length - 1):
                    decode_req.reset(InferencePhase.DECODE)
                    decode_req.input_token_ids.append(first_token)
                    decode_req.start_position += 1
                    self.service.batcher.submit(decode_req)
                    await decode_req.done
                    next_token = np.argmax(decode_req.result_logits.items)
                    result_sequence.append(next_token)
                    first_token = next_token

                batch_results.append(result_sequence)
                decode_req.free_cache_pages()

            self.results[batch_size] = batch_results

            first_result = batch_results[0]
            for result in batch_results[1:]:
                assert np.array_equal(
                    first_result, result
                ), f"Inconsistent results within batch size {batch_size}"

        first_batch_result = self.results[self.batch_sizes[0]][0]
        for batch_size in self.batch_sizes[1:]:
            assert np.array_equal(
                first_batch_result, self.results[batch_size][0]
            ), f"Inconsistent results between batch sizes {self.batch_sizes[0]} and {batch_size}"


def test_batch_and_nobatch_consistency(server_instance):
    """Test that requests produce identical results regardless of batch size."""
    # Test parameters
    input_tokens = [1, 2, 3, 4]  # Initial sequence
    batch_sizes = [1, 2, 4]  # Different batch sizes to test
    max_response_length = 3  # Number of decode steps

    with server_instance.start_service_only() as generate_service:
        # Create and run the test process
        test_process = BatchConsistencyTestProcess(
            generate_service, input_tokens, batch_sizes, max_response_length
        )
        test_process.launch()
