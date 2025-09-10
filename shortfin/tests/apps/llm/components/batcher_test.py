import asyncio
from uuid import uuid4
import pytest

import shortfin.array as sfnp

from unittest.mock import AsyncMock, MagicMock, patch

from shortfin import ProgramIsolation

from shortfin_apps.llm.components.batching.modes.default import (
    LlmBatcherProcess,
    PrefillTaskResponder,
)
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.invocation import (
    PrefillTask,
    LlmInvocationProcess,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)


@pytest.fixture
def model_params():
    return ModelParams(
        max_seq_len=42,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        has_prefill_position=False,
        decode_batch_sizes=[4],
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=42,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture
def llm_batcher_process(model_params, fiber, cache):
    return LlmBatcherProcess(
        name="test-batcher",
        fiber=fiber,
        page_cache=cache,
        model_params=model_params,
        functions=None,
        ideal_batch_size=4,
        program_isolation=ProgramIsolation.PER_CALL.value,
        llm_task_responder=PrefillTaskResponder(),
    )


class MockVoidFuture:
    def __init__(self):
        self._event = asyncio.Event()

    def set_success(self):
        self._event.set()

    def __await__(self):
        return self._event.wait().__await__()


@pytest.fixture()
def exec_req_list():
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        input_tokens = [0, 1, 2, 3, 4, 5]

        exec_reqs = []
        for _ in range(4):
            exec_req = LlmInferenceExecRequest(
                phase=InferencePhase.PREFILL,
                input_token_ids=input_tokens,
                rid=str(uuid4()),
            )
            exec_reqs.append(exec_req)
            input_tokens = [val + 1 for val in input_tokens]

        yield exec_reqs


@pytest.fixture
def prefill_task(device_array_cache, exec_req_list, model_params):
    return PrefillTask(
        exec_requests=exec_req_list,
        array_cache=device_array_cache,
        seq_stride=model_params.paged_kv_cache.block_seq_stride,
    )


@pytest.fixture
def llm_invoker(model_params, fiber, device_array_cache):
    return LlmInvocationProcess(
        name="test-executor",
        fiber=fiber,
        array_cache=device_array_cache,
        functions=None,
        seq_stride=42,
        page_tables=None,
        program_isolation=ProgramIsolation.PER_CALL.value,
    )


class TestLlmBatcherProcess:
    @pytest.mark.asyncio
    async def test_board_flights(
        self, llm_batcher_process: LlmBatcherProcess, exec_req_list
    ):
        llm_batcher_process.board = MagicMock()

        ## Empty
        llm_batcher_process.pending = set()
        await llm_batcher_process.board_flights()
        assert llm_batcher_process.board.call_count == 0
        llm_batcher_process.board.reset_mock()

        assert llm_batcher_process.pending == set()

        ## Non-empty
        to_schedule = set(exec_req_list)
        llm_batcher_process.pending = to_schedule
        await llm_batcher_process.board_flights()

        assert llm_batcher_process.board.call_count == 1
        call_args = llm_batcher_process.board.call_args.args
        assert len(call_args) == 3
        assert call_args[0] == llm_batcher_process.page_cache
        assert call_args[1] == llm_batcher_process.fiber
        assert set(call_args[2]) == set(exec_req_list)

        assert llm_batcher_process.pending == set()
