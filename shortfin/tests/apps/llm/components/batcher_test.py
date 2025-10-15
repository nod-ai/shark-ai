import asyncio
import math
import pytest

import shortfin.array as sfnp

from random import randint
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from shortfin import ProgramIsolation

from shortfin_apps.llm.components.batching.modes.default import (
    LlmBatcherProcess,
    PrefillBatcherProcess,
    PrefillTaskResponder,
)
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.invocation import (
    LlmInvocationProcess,
    LlmTaskInput,
    PrefillTask,
)
from shortfin_apps.llm.components.kvcache.attention_cache_abstract import (
    CacheInfo,
)
from shortfin_apps.llm.components.kvcache.page_pool import (
    PageInfo,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.scheduler import Scheduler


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
            block_seq_stride=2,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture
def llm_batcher_process(model_params, fiber, cache):
    ideal_batch_size = 4
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)
    return LlmBatcherProcess(
        name="test-batcher",
        fiber=fiber,
        page_cache=cache,
        model_params=model_params,
        functions=None,
        ideal_batch_size=4,
        program_isolation=ProgramIsolation.PER_CALL.value,
        scheduler=scheduler,
        llm_task_responder=PrefillTaskResponder(scheduler=scheduler),
    )


@pytest.fixture(scope="function")
def prefill_batcher_process(model_params, fiber, cache):
    ideal_batch_size = 4
    return PrefillBatcherProcess(
        fiber=fiber,
        page_cache=cache,
        model_params=model_params,
        prefill_functions={ideal_batch_size: AsyncMock()},
        program_isolation=ProgramIsolation.PER_CALL.value,
        chunk_block_size=None,
    )


@pytest.fixture(scope="function")
def prefill_batcher_process_chunked(model_params, fiber, cache):
    ideal_batch_size = 4
    return PrefillBatcherProcess(
        fiber=fiber,
        page_cache=cache,
        model_params=model_params,
        prefill_functions={ideal_batch_size: AsyncMock()},
        program_isolation=ProgramIsolation.PER_CALL.value,
        chunk_block_size=2,
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
        input_tokens = [0, 1, 2, 3, 4, 5, 6, 7]

        exec_reqs = []
        for _ in range(4):
            exec_req = LlmInferenceExecRequest(
                phase=InferencePhase.PREFILL,
                input_token_ids=input_tokens,
                rid=str(uuid4()),
            )
            exec_reqs.append(exec_req)
            input_tokens = [val + 1 for val in input_tokens]

        # --- Assign unique page_ids for each request ---
        page_seq_stride = 2
        next_page_id = 0
        for req in exec_reqs:
            num_pages = math.ceil(len(req.input_token_ids) / page_seq_stride)
            req.page_ids = list(range(next_page_id, next_page_id + num_pages))
            next_page_id += num_pages

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


def _get_task_input(exec_req):
    return LlmTaskInput(
        rid=exec_req.orig_instance_id,
        instance_id=exec_req.instance_id,
        block_count=exec_req.block_count,
        input_tokens=tuple(exec_req.input_token_ids),
        seq_len=len(exec_req.input_token_ids),
        page_ids=tuple(exec_req.page_ids),
        start_position=exec_req.start_position,
    )


class TestLlmBatcherProcess:
    @pytest.mark.asyncio
    async def test_board_flights(
        self, llm_batcher_process: LlmBatcherProcess, exec_req_list
    ):
        llm_batcher_process.board = MagicMock()

        ## Empty
        await llm_batcher_process.board_flights()
        assert llm_batcher_process.board.call_count == 0
        llm_batcher_process.board.reset_mock()

        ## Non-empty
        task_inputs = []
        for req in exec_req_list:
            task_input = _get_task_input(req)
            task_inputs.append(task_input)
            llm_batcher_process.scheduler.schedule_job(task_input)

        await llm_batcher_process.board_flights()

        assert llm_batcher_process.board.call_count == 1
        call_args = llm_batcher_process.board.call_args.args
        assert len(call_args) == 3
        assert call_args[0] == llm_batcher_process.page_cache
        assert call_args[1] == llm_batcher_process.fiber
        assert set(call_args[2]) == set(task_inputs)


class TestPrefillBatcherProcess:
    def test_handle_inference_request(
        self, prefill_batcher_process_chunked: PrefillBatcherProcess, exec_req_list
    ):
        req = exec_req_list[0]
        with patch.object(
            prefill_batcher_process_chunked.scheduler,
            "schedule_job",
        ) as mock_schedule_job:
            prefill_batcher_process_chunked.handle_inference_request(req)
            assert mock_schedule_job.call_count == 2

    def test_make_task_inputs_no_chunking(
        self, prefill_batcher_process: PrefillBatcherProcess, exec_req_list
    ):
        for req in exec_req_list:
            task_input = prefill_batcher_process.make_task_inputs(req)
            expected_task_input = _get_task_input(req)
            assert task_input[0] == expected_task_input

    def test_make_task_inputs_chunked_single(
        self, prefill_batcher_process_chunked: PrefillBatcherProcess, exec_req_list
    ):
        # First in one chunk
        req = exec_req_list[0]
        req.input_token_ids = req.input_token_ids[:4]
        req.page_ids = req.page_ids[:2]
        task_inputs = prefill_batcher_process_chunked.make_task_inputs(req)
        expected = [
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=2,
                input_tokens=(0, 1, 2, 3),
                seq_len=4,
                page_ids=(0, 1),
                start_position=0,
            ),
        ]

        assert len(task_inputs) == 1
        assert task_inputs == expected

    def test_make_task_inputs_chunked_multiple(
        self, prefill_batcher_process_chunked: PrefillBatcherProcess, exec_req_list
    ):
        # Full Chunks
        req = exec_req_list[0]
        task_inputs = prefill_batcher_process_chunked.make_task_inputs(req)
        expected = [
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=2,
                input_tokens=(0, 1, 2, 3),
                seq_len=4,
                page_ids=(0, 1),
                start_position=0,
            ),
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=4,
                input_tokens=(4, 5, 6, 7),
                seq_len=8,
                page_ids=(0, 1, 2, 3),
                start_position=4,
            ),
        ]

        assert len(task_inputs) == 2
        assert task_inputs == expected

        # Partial Chunk
        req.input_token_ids.append(8)
        req.page_ids.append(4)
        task_inputs = prefill_batcher_process_chunked.make_task_inputs(req)
        expected = [
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=2,
                input_tokens=(0, 1, 2, 3),
                seq_len=4,
                page_ids=(0, 1),
                start_position=0,
            ),
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=4,
                input_tokens=(4, 5, 6, 7),
                seq_len=8,
                page_ids=(0, 1, 2, 3),
                start_position=4,
            ),
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=5,
                input_tokens=(8,),
                seq_len=9,
                page_ids=(0, 1, 2, 3, 4),
                start_position=8,
            ),
        ]

        assert len(task_inputs) == 3
        assert task_inputs == expected

    def test_make_task_inputs_varied_start_positions(
        self, prefill_batcher_process_chunked, exec_req_list
    ):
        for i in range(len(exec_req_list)):
            req = exec_req_list[i]
            req.start_position = i * 2

        task_inputs = []
        for req in exec_req_list:
            task_inputs.append(prefill_batcher_process_chunked.make_task_inputs(req))

        # Start positions is zero, so should have full chunks from beginning
        assert task_inputs[0] == [
            LlmTaskInput(
                rid=exec_req_list[0].orig_instance_id,
                instance_id=exec_req_list[0].instance_id,
                block_count=2,
                input_tokens=(0, 1, 2, 3),
                seq_len=4,
                page_ids=(0, 1),
                start_position=0,
            ),
            LlmTaskInput(
                rid=exec_req_list[0].orig_instance_id,
                instance_id=exec_req_list[0].instance_id,
                block_count=4,
                input_tokens=(4, 5, 6, 7),
                seq_len=8,
                page_ids=(0, 1, 2, 3),
                start_position=4,
            ),
        ]
        # Start position is 2, so first chunk should start from 3rd token
        assert task_inputs[1] == [
            LlmTaskInput(
                rid=exec_req_list[1].orig_instance_id,
                instance_id=exec_req_list[1].instance_id,
                block_count=3,
                input_tokens=(3, 4, 5, 6),
                seq_len=6,
                page_ids=(4, 5, 6),
                start_position=2,
            ),
            LlmTaskInput(
                rid=exec_req_list[1].orig_instance_id,
                instance_id=exec_req_list[1].instance_id,
                block_count=4,
                input_tokens=(7, 8),
                seq_len=8,
                page_ids=(4, 5, 6, 7),
                start_position=6,
            ),
        ]
        # Start position is 4, so first chunk should start from 5th token
        assert task_inputs[2] == [
            LlmTaskInput(
                rid=exec_req_list[2].orig_instance_id,
                instance_id=exec_req_list[2].instance_id,
                block_count=4,
                input_tokens=(6, 7, 8, 9),
                seq_len=8,
                page_ids=(8, 9, 10, 11),
                start_position=4,
            ),
        ]
        # Start position is 6, so first chunk should start from 7th token
        assert task_inputs[3] == [
            LlmTaskInput(
                rid=exec_req_list[3].orig_instance_id,
                instance_id=exec_req_list[3].instance_id,
                block_count=4,
                input_tokens=(9, 10),
                seq_len=8,
                page_ids=(12, 13, 14, 15),
                start_position=6,
            ),
        ]
