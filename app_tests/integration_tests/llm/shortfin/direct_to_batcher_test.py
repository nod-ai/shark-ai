import pytest
import numpy as np
import asyncio
import shortfin as sf

from app_tests.integration_tests.llm.server_management import (
    ServerInstance,
    ServerConfig,
)
from app_tests.integration_tests.llm.model_management import (
    ModelProcessor,
    ModelConfig,
)
from app_tests.integration_tests.llm.device_settings import CPU
from shortfin_apps.llm.components.messages import (
    InferencePhase,
    LlmInferenceExecRequest,
)


pytestmark = pytest.mark.parametrize(
    "model_artifacts,generate_service",
    [
        (ModelConfig.get(name="tinystories_llama2_25m"), {"prefix_sharing": "none"}),
    ],
    ids=[
        "tinystories_llama2_25m_none",
    ],
    indirect=True,
)


class BatchConsistencyTestProcess(sf.Process):
    """Process to test consistency of results across different batch sizes.

    This is necessary because InferenceExecRequest uses shortfin.VoidFuture
    which can only be created on a process (which belongs to a fiber that a worker works through).
    """

    def __init__(self, service, input_tokens, batch_sizes, max_response_length):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.input_tokens = input_tokens
        self.batch_sizes = batch_sizes
        self.max_response_length = max_response_length
        self.results = {}  # Store results for each batch size
        # TODO: modify the batcher to guarantee the batch we send isn't split by strobe messages

    async def run(self):
        for batch_size in self.batch_sizes:
            batch_results = []
            for _ in range(batch_size):
                prefill_req = LlmInferenceExecRequest(
                    phase=InferencePhase.PREFILL,
                    input_token_ids=self.input_tokens,
                    rid=f"test-{batch_size}",
                )
                prefill_req.return_host_array = True
                self.service.prefill_batcher.submit(prefill_req)
                await prefill_req.done
                first_token = np.argmax(prefill_req.result_logits.items)
                result_sequence = [first_token]

                decode_req = prefill_req
                for _ in range(self.max_response_length - 1):
                    decode_req.reset(InferencePhase.DECODE)
                    decode_req.input_token_ids.append(first_token)
                    decode_req.start_position += 1
                    self.service.decode_batcher.submit(decode_req)
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


def test_batch_and_nobatch_consistency(model_artifacts, generate_service):
    """
    Test that requests produce identical results regardless of batch size.

    If this test fails, it means that changing the batch size changes the generation results.

    Look for kvcache corruption due to
    - improper seq_len / current_position handling in service.py
    - improper masking in amdsharktank
    """
    # Create and run the test process
    test_process = BatchConsistencyTestProcess(
        generate_service,
        input_tokens=[1, 2, 3, 4],
        batch_sizes=[1, 2, 3, 4],
        max_response_length=3,
    )
    test_process.launch()
