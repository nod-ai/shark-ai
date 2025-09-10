import asyncio
import logging
import math
import pytest

import shortfin as sf
import shortfin.array as sfnp

from random import randint
from typing import List, Union
from unittest.mock import MagicMock, patch
from uuid import uuid4

from shortfin_apps.llm.components.batching.modes.default import (
    DecodeTaskResponder,
    PrefillTaskResponder,
)
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.device_array_cache import (
    Allocation,
    WrappedAllocation,
)
from shortfin_apps.llm.components.invocation import (
    LlmInvocationProcess,
    LlmTaskInput,
    LlmTaskResponder,
    PrefillTask,
    DecodeTask,
    _pad_list,
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


logger = logging.getLogger(__name__)


class MockVoidFuture:
    def __init__(self):
        self._event = asyncio.Event()

    def set_success(self):
        self._event.set()

    def __await__(self):
        return self._event.wait().__await__()


class DummyDeviceArrayAllocation:
    def __init__(self, device_array: sfnp.device_array):
        self.device = device_array
        self.shape = device_array.shape
        self.released = False

    def release(self):
        self.released = True


@pytest.fixture
def model_params():
    return ModelParams(
        max_seq_len=42,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[4],
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=42,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture(scope="function")
def staggered_exec_req_list(cache_ref_count, page_pool):
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        exec_reqs = []
        for _ in range(4):
            input_tokens = [_ for _ in range(randint(2, 10))]
            exec_req = LlmInferenceExecRequest(
                phase=InferencePhase.PREFILL,
                input_token_ids=input_tokens,
                rid=str(uuid4()),
            )
            exec_reqs.append(exec_req)
            input_tokens = [val + 1 for val in input_tokens]

        page_offset = 0
        for req in exec_reqs:
            req._cache = cache_ref_count
            pages = [
                PageInfo(index=page_offset + i, pool=page_pool)
                for i in range(len(req.input_token_ids) // 2 + 1)
            ]
            req.allocated_cache_info = CacheInfo(
                num_tokens=len(req.input_token_ids),
                pages=pages,
                pool=page_pool,
            )
            req.page_ids = [page.index for page in pages]
            page_offset += len(pages)

        yield exec_reqs


def _get_task_inputs(
    exec_requests: List[LlmInferenceExecRequest], prefill_w_start_pos=False
) -> List[LlmTaskInput]:
    task_inputs = []
    for req in exec_requests:
        task_inputs.append(
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=req.block_count,
                seq_stride=2,
                input_tokens=req.input_token_ids,
                page_ids=req.page_ids,
                start_position=req.start_position,
            )
        )

    return task_inputs


@pytest.fixture(scope="function")
def prefill_task(staggered_exec_req_list, device_array_cache, page_pool) -> PrefillTask:
    """Fixture to create an instance of LlmTask."""
    page_tables = page_pool.acquire_free_pages(len(staggered_exec_req_list))
    task_input = _get_task_inputs(staggered_exec_req_list)
    return PrefillTask(
        task_inputs=task_input,
        array_cache=device_array_cache,
        page_tables=page_tables,
        has_prefill_position=False,
    )


@pytest.fixture(scope="function")
def prefill_task_w_start_pos(
    staggered_exec_req_list, device_array_cache, page_pool
) -> PrefillTask:
    """Fixture to create an instance of LlmTask."""
    page_tables = page_pool.acquire_free_pages(len(staggered_exec_req_list))
    task_input = _get_task_inputs(staggered_exec_req_list, True)
    return PrefillTask(
        task_inputs=task_input,
        array_cache=device_array_cache,
        page_tables=page_tables,
        has_prefill_position=True,
    )


@pytest.fixture(scope="function")
def decode_task(staggered_exec_req_list, device_array_cache, page_pool) -> DecodeTask:
    for req in staggered_exec_req_list:
        req.phase = InferencePhase.DECODE
        req.start_position = len(req.input_token_ids) - 1

    page_tables = page_pool.acquire_free_pages(len(staggered_exec_req_list))
    task_inputs = _get_task_inputs(staggered_exec_req_list)
    return DecodeTask(
        task_inputs=task_inputs,
        array_cache=device_array_cache,
        page_tables=page_tables,
    )


def _get_batch_seq_len(
    exec_requests: List[LlmInferenceExecRequest], seq_stride: int
) -> int:
    max_len = max(len(req.input_token_ids) for req in exec_requests)
    return int(math.ceil(max_len / seq_stride) * 2)


@pytest.fixture(scope="function")
def result_logits_none_indices(staggered_exec_req_list, fiber):
    """Fixture to create a result logits device array."""
    vocab_size = 16
    batch_size = len(staggered_exec_req_list)
    seq_len = _get_batch_seq_len(staggered_exec_req_list, seq_stride=2)

    logits = sfnp.device_array(
        fiber.device(0), [batch_size, seq_len, vocab_size], dtype=sfnp.float16
    )

    # Prepare one flat buffer, zero-initialized
    total = batch_size * seq_len * vocab_size
    flat = [0] * total

    # Helper: flatten (i, t, v) -> offset
    def offset(i, t, v):
        return ((i * seq_len) + t) * vocab_size + v

    # Fill recognizable pattern: for each batch i, at timestep sl,
    # set logits[i, sl, 0:sl] = [0, 1, ..., sl-1]
    for i, req in enumerate(staggered_exec_req_list):
        sl = len(req.input_token_ids) - 1
        if sl <= 0:
            continue
        upto = min(sl, vocab_size)
        for v in range(upto):
            flat[offset(i, sl, v)] = v

    # Single map/write
    with logits.map(discard=True) as m:
        m.items = flat  # one bulk write

    return logits, None  # indices


@pytest.fixture(scope="function")
def result_logits_none_indices_decode(staggered_exec_req_list, fiber):
    """Fixture to create a result logits device array."""
    vocab_size = 16
    batch_size = len(staggered_exec_req_list)
    seq_len = _get_batch_seq_len(staggered_exec_req_list, seq_stride=2)

    logits = sfnp.device_array(
        fiber.device(0), [batch_size, seq_len, vocab_size], dtype=sfnp.float16
    )

    # Flat zero-initialized buffer for the whole tensor
    total = batch_size * seq_len * vocab_size
    flat = [0] * total

    # Helper to compute flat index for (batch i, timestep t, vocab v)
    def offset(i, t, v):
        return ((i * seq_len) + t) * vocab_size + v

    # Fill recognizable pattern at t=0 for every batch:
    # logits[i, 0, :] = [0, 1, 2, ..., vocab_size-1]
    if seq_len > 0:
        for i in range(batch_size):
            for v in range(vocab_size):
                flat[offset(i, 0, v)] = v

    # Single map/write
    with logits.map(discard=True) as m:
        m.items = flat

    return logits, None  # indices


@pytest.fixture(scope="function")
def result_logits_w_indices(staggered_exec_req_list, fiber):
    """Fixture to create a result logits device array with indices."""
    batch_size = len(staggered_exec_req_list)
    seq_len = _get_batch_seq_len(staggered_exec_req_list, seq_stride=2)
    k = 4
    device0 = fiber.device(0)

    logits = sfnp.device_array(device0, [batch_size, seq_len, k], dtype=sfnp.float16)
    indices = sfnp.device_array(device0, [batch_size, seq_len, k], dtype=sfnp.int32)

    # Flat zero-filled buffers
    logits_flat = [0] * (batch_size * seq_len * k)
    indices_flat = [0] * (batch_size * seq_len * k)

    # Helper to get flat index
    def offset(i, t, v):
        return ((i * seq_len) + t) * k + v

    # Populate pattern
    for i, req in enumerate(staggered_exec_req_list):
        sl = len(req.input_token_ids) - 1
        for v in range(k):
            logits_flat[offset(i, sl, v)] = i + v
            indices_flat[offset(i, sl, v)] = 10 + i + v

    # Single map for logits
    with logits.map(discard=True) as m:
        m.items = logits_flat

    # Single map for indices
    with indices.map(discard=True) as m:
        m.items = indices_flat

    return logits, indices


@pytest.fixture(scope="function")
def result_logits_w_indices_decode(staggered_exec_req_list, fiber):
    """Fixture to create a result logits device array with indices."""
    batch_size = len(staggered_exec_req_list)
    seq_len = _get_batch_seq_len(staggered_exec_req_list, seq_stride=2)
    k = 4
    device0 = fiber.device(0)

    logits = sfnp.device_array(device0, [batch_size, seq_len, k], dtype=sfnp.float16)
    indices = sfnp.device_array(device0, [batch_size, seq_len, k], dtype=sfnp.int32)

    # Zero-initialized flat buffers
    logits_flat = [0] * (batch_size * seq_len * k)
    indices_flat = [0] * (batch_size * seq_len * k)

    # Helper to compute flat index
    def offset(i, t, v):
        return ((i * seq_len) + t) * k + v

    # Fill pattern at t = 0 for each batch
    if seq_len > 0:
        for i in range(batch_size):
            for v in range(k):
                logits_flat[offset(i, 0, v)] = i + v
                indices_flat[offset(i, 0, v)] = 10 + i + v

    # Single map for logits
    with logits.map(discard=True) as m:
        m.items = logits_flat

    # Single map for indices
    with indices.map(discard=True) as m:
        m.items = indices_flat

    return logits, indices


@pytest.fixture(scope="function")
def decode_task_responder():
    return DecodeTaskResponder()


@pytest.fixture(scope="function")
def prefill_task_responder():
    return PrefillTaskResponder()


@pytest.fixture(scope="function")
def llm_invoker(prefill_task: PrefillTask, fiber):
    async def invocation_fn(*args, fiber=None):
        return tuple(args)

    mock_responder = MagicMock(spec=LlmTaskResponder)
    return LlmInvocationProcess(
        name="test-invoker",
        fiber=fiber,
        llm_task=prefill_task,
        functions={},
        program_isolation=sf.ProgramIsolation.PER_CALL,
        responder=mock_responder,
    )


def _validate_prefill_args(
    exec_reqs: List[LlmInferenceExecRequest],
    args: List[Union[Allocation, WrappedAllocation]],
):
    tokens, seq_lens, seq_block_ids = [arg.host.items.tolist() for arg in args[:3]]
    block_count = max(req.block_count for req in exec_reqs)
    batch_seq_len = _get_batch_seq_len(exec_reqs, seq_stride=2)
    assert len(tokens) % batch_seq_len == 0
    for i, req in enumerate(exec_reqs):
        offset = i * batch_seq_len
        results = tokens[offset : offset + batch_seq_len]
        expected = _pad_list(
            req.input_token_ids,
            batch_seq_len,
        )
        assert results == expected

    assert seq_lens == [len(req.input_token_ids) for req in exec_reqs]

    for i, req in enumerate(exec_reqs):
        offset = i * block_count
        results = seq_block_ids[offset : offset + block_count]

        block_ids = req.cache_page_indices(batch_seq_len)
        expected = _pad_list(
            block_ids,
            block_count,
        )

        assert results == expected


def _validate_prefill_args_w_start_pos(
    exec_reqs: List[LlmInferenceExecRequest],
    args: List[Union[Allocation, WrappedAllocation]],
):
    tokens, start_positions, seq_lens, seq_block_ids = [
        arg.host.items.tolist() for arg in args[:4]
    ]
    block_count = max(req.block_count for req in exec_reqs)
    batch_seq_len = _get_batch_seq_len(exec_reqs, seq_stride=2)
    assert len(tokens) % batch_seq_len == 0
    for i, req in enumerate(exec_reqs):
        offset = i * batch_seq_len
        results = tokens[offset : offset + batch_seq_len]
        expected = _pad_list(
            req.input_token_ids,
            batch_seq_len,
        )
        assert results == expected

    for i, req in enumerate(exec_reqs):
        assert start_positions[i] == req.start_position

    assert seq_lens == [len(req.input_token_ids) for req in exec_reqs]

    for i, req in enumerate(exec_reqs):
        offset = i * block_count
        results = seq_block_ids[offset : offset + block_count]

        block_ids = req.cache_page_indices(batch_seq_len)
        expected = _pad_list(
            block_ids,
            block_count,
        )

        assert results == expected


class TestPrefillTask:
    def test_get_args(self, lsys, prefill_task: PrefillTask, staggered_exec_req_list):
        async def _test():
            args = await prefill_task.prepare_args(
                batch_size=prefill_task.req_count,
            )

            assert all(isinstance(arg, Allocation) for arg in args[:3])
            assert all(isinstance(arg, WrappedAllocation) for arg in args[3:])

            _validate_prefill_args(
                exec_reqs=staggered_exec_req_list,
                args=args,
            )

        lsys.run(_test())

    def test_process_results(
        self,
        fiber,
        lsys,
        prefill_task: PrefillTask,
        prefill_task_responder: PrefillTaskResponder,
        result_logits_none_indices,
        staggered_exec_req_list,
    ):
        async def _test():
            device0 = fiber.device(0)
            for req in staggered_exec_req_list:
                prefill_task_responder.add_request(req)
            args = await prefill_task.prepare_args(
                batch_size=prefill_task.req_count,
            )

            logits, _ = result_logits_none_indices
            vocab_size = logits.shape[-1]
            logits, indices = await prefill_task.process_results(
                args=args,
                logits=logits,
                indices=None,
                device0=device0,
            )

            prefill_task_responder.set_success(
                prefill_task,
                logits,
                indices,
            )

            # Verify that the logits were processed correctly
            for req in staggered_exec_req_list:
                sl = len(req.input_token_ids) - 1
                expected = _pad_list(
                    [i for i in range(sl)],
                    vocab_size,
                )
                result = req.result_logits.items.tolist()
                assert result == expected

        lsys.run(_test())

    def test_process_results_w_indices(
        self,
        fiber,
        lsys,
        prefill_task: PrefillTask,
        prefill_task_responder: PrefillTaskResponder,
        result_logits_w_indices,
        staggered_exec_req_list,
    ):
        async def _test():
            device0 = fiber.device(0)
            for req in staggered_exec_req_list:
                prefill_task_responder.add_request(req)
            args = await prefill_task.prepare_args(
                batch_size=prefill_task.req_count,
            )

            logits, indices = result_logits_w_indices
            logits, indices = await prefill_task.process_results(
                args=args,
                logits=logits,
                indices=indices,
                device0=device0,
            )

            prefill_task_responder.set_success(prefill_task, logits, indices)

            # Verify that the logits were processed correctly
            for i, req in enumerate(staggered_exec_req_list):
                assert req.result_logits.items.tolist() == [i, i + 1, i + 2, i + 3]
                assert req.result_indices.items.tolist() == [
                    10 + i,
                    11 + i,
                    12 + i,
                    13 + i,
                ]

        lsys.run(_test())


class TestPrefillTaskWithStartPos:
    def test_get_args(
        self, lsys, prefill_task_w_start_pos: PrefillTask, staggered_exec_req_list
    ):
        async def _test():
            args = await prefill_task_w_start_pos.prepare_args(
                batch_size=prefill_task_w_start_pos.req_count,
            )

            assert all(isinstance(arg, Allocation) for arg in args[:4])
            assert all(isinstance(arg, WrappedAllocation) for arg in args[4:])

            _validate_prefill_args_w_start_pos(
                exec_reqs=staggered_exec_req_list,
                args=args,
            )

        lsys.run(_test())


def _validate_decode_args(
    exec_reqs: List[LlmInferenceExecRequest],
    args: List[Union[Allocation, WrappedAllocation]],
):
    block_count = max(req.block_count for req in exec_reqs)
    tokens, seq_lens, start_positions, seq_block_ids = [
        arg.host.items.tolist() for arg in args[:4]
    ]

    for i, req in enumerate(exec_reqs):
        assert tokens[i] == req.input_token_ids[-1]

    assert seq_lens == [req.start_position + 1 for req in exec_reqs]
    assert start_positions == [req.start_position for req in exec_reqs]

    batch_seq_len = _get_batch_seq_len(exec_reqs, seq_stride=2)
    for i, req in enumerate(exec_reqs):
        offset = i * block_count
        results = seq_block_ids[offset : offset + block_count]

        # mirror get_args_data logic
        block_ids = req.cache_page_indices(batch_seq_len)
        expected = _pad_list(block_ids, block_count)
        assert results == expected


class TestDecodeTask:
    def test_get_args(self, lsys, decode_task: DecodeTask, staggered_exec_req_list):
        async def _test():
            args = await decode_task.prepare_args(
                batch_size=decode_task.req_count,
            )

            assert all(isinstance(arg, Allocation) for arg in args[:4])
            assert all(isinstance(arg, WrappedAllocation) for arg in args[4:])

            _validate_decode_args(
                exec_reqs=staggered_exec_req_list,
                args=args,
            )

        lsys.run(_test())

    def test_process_results(
        self,
        fiber,
        lsys,
        decode_task,
        decode_task_responder,
        result_logits_none_indices_decode,
        staggered_exec_req_list,
    ):
        async def _test():
            device0 = fiber.device(0)
            for req in staggered_exec_req_list:
                decode_task_responder.add_request(req)
            logits, _ = result_logits_none_indices_decode
            vocab_size = logits.shape[-1]
            args = await decode_task.prepare_args(
                batch_size=decode_task.req_count,
            )

            logits, indices = await decode_task.process_results(
                args=args,
                logits=logits,
                indices=None,
                device0=device0,
            )

            decode_task_responder.set_success(decode_task, logits, indices)

            for req in staggered_exec_req_list:
                results = req.result_logits.items.tolist()

                assert results == [_ for _ in range(vocab_size)]

                assert (
                    req.result_indices is None
                ), "Indices should be None for decode task"

        lsys.run(_test())

    def test_process_results_w_indices(
        self,
        fiber,
        lsys,
        decode_task,
        decode_task_responder,
        result_logits_w_indices_decode,
        staggered_exec_req_list,
    ):
        async def _test():
            device0 = fiber.device(0)
            for req in staggered_exec_req_list:
                decode_task_responder.add_request(req)
            args = await decode_task.prepare_args(
                batch_size=decode_task.req_count,
            )

            logits, indices = result_logits_w_indices_decode
            logits, indices = await decode_task.process_results(
                args=args,
                logits=logits,
                indices=indices,
                device0=device0,
            )

            decode_task_responder.set_success(decode_task, logits, indices)

            # Verify get_result picked the exact [i, sl, :] vectors
            for i, req in enumerate(staggered_exec_req_list):
                assert req.result_logits.items.tolist() == [i, i + 1, i + 2, i + 3]
                assert req.result_indices.items.tolist() == [
                    10 + i,
                    11 + i,
                    12 + i,
                    13 + i,
                ]

        lsys.run(_test())


class TestLlmInvocationProcess:
    def test_run_none_indices(
        self,
        lsys,
        llm_invoker: LlmInvocationProcess,
        prefill_task,
        prefill_task_responder: PrefillTaskResponder,
        result_logits_none_indices,
        staggered_exec_req_list,
    ):
        async def _test():
            async def entrypoint(*args, fiber=None):
                return result_logits_none_indices

            for req in staggered_exec_req_list:
                prefill_task_responder.add_request(req)

            llm_invoker._functions = {prefill_task.req_count: entrypoint}
            llm_invoker._responder = prefill_task_responder
            await llm_invoker.run()

            logits, _ = result_logits_none_indices
            vocab_size = logits.shape[-1]
            for req in staggered_exec_req_list:
                seq_len = len(req.input_token_ids) - 1

                expected = _pad_list(
                    [_ for _ in range(seq_len)],
                    vocab_size,
                )
                results = req.result_logits.items.tolist()

                assert results == expected

        lsys.run(_test())
