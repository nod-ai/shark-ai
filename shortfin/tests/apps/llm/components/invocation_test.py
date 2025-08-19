import asyncio
import logging
import pytest

import shortfin as sf
import shortfin.array as sfnp

from random import randint
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from shortfin_apps.llm.components.device_array_cache import (
    Allocation,
    WrappedAllocation,
)
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.invocation import (
    LlmInvoker,
    PrefillTask,
    DecodeTask,
    _pad_list,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.kvcache.page_pool import (
    PageInfo,
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
            allocation = BasePagedAttentionCacheAllocation(pages, cache=cache_ref_count)
            req.allocation = allocation
            page_offset += len(pages)

        yield exec_reqs


@pytest.fixture(scope="function")
def prefill_task(staggered_exec_req_list, device_array_cache) -> PrefillTask:
    """Fixture to create an instance of LlmTask."""
    return PrefillTask(
        exec_requests=staggered_exec_req_list,
        array_cache=device_array_cache,
        seq_stride=2,
    )


@pytest.fixture(scope="function")
def decode_task(staggered_exec_req_list, device_array_cache) -> DecodeTask:
    for req in staggered_exec_req_list:
        req.phase = InferencePhase.DECODE
        req.start_position = len(req.input_token_ids) - 1

    return DecodeTask(
        exec_requests=staggered_exec_req_list,
        array_cache=device_array_cache,
        seq_stride=2,
    )


@pytest.fixture(scope="function")
def result_logits_none_indices(prefill_task, fiber):
    """Fixture to create a result logits device array."""
    vocab_size = 16
    batch_size = len(prefill_task.exec_requests)
    seq_len = max(len(req.input_token_ids) for req in prefill_task.exec_requests)

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
    for i, req in enumerate(prefill_task.exec_requests):
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
def result_logits_none_indices_decode(decode_task, fiber):
    """Fixture to create a result logits device array."""
    vocab_size = 16
    batch_size = len(decode_task.exec_requests)
    seq_len = max(len(req.input_token_ids) for req in decode_task.exec_requests)

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
def result_logits_w_indices(prefill_task, fiber):
    """Fixture to create a result logits device array with indices."""
    batch_size = len(prefill_task.exec_requests)
    seq_len = max(len(req.input_token_ids) for req in prefill_task.exec_requests)
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
    for i, req in enumerate(prefill_task.exec_requests):
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
def result_logits_w_indices_decode(prefill_task, fiber):
    """Fixture to create a result logits device array with indices."""
    batch_size = len(prefill_task.exec_requests)
    seq_len = max(len(req.input_token_ids) for req in prefill_task.exec_requests)
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
def llm_invoker(prefill_task: PrefillTask, fiber, device_array_cache, page_pool):
    async def invocation_fn(*args, fiber=None):
        return tuple(args)

    return LlmInvoker(
        name="test-invoker",
        fiber=fiber,
        array_cache=device_array_cache,
        llm_task=prefill_task,
        functions={},
        seq_stride=prefill_task._seq_stride,
        page_tables=page_pool.page_tables,
        program_isolation=sf.ProgramIsolation.PER_CALL,
    )


class TestPrefillTask:
    def test_get_args_data(self, prefill_task):
        block_count = max(
            len(req.allocation.pages) for req in prefill_task.exec_requests
        )
        batch_seq_len = max(
            len(req.input_token_ids) for req in prefill_task.exec_requests
        )

        (token_vals, seq_len_vals, seq_block_ids_vals,) = prefill_task.get_args_data(
            exec_requests=prefill_task.exec_requests,
            batch_seq_len=batch_seq_len,
            block_count=block_count,
        )

        max_token_len = max(
            len(req.input_token_ids) for req in prefill_task.exec_requests
        )
        for i, req in enumerate(prefill_task.exec_requests):
            offset = i * max_token_len
            results = token_vals[offset : offset + max_token_len]
            expected = req.input_token_ids + [0] * (
                max_token_len - len(req.input_token_ids)
            )
            assert results == expected

        assert seq_len_vals == [
            len(req.input_token_ids) for req in prefill_task.exec_requests
        ]

        for i, req in enumerate(prefill_task.exec_requests):
            offset = i * block_count
            results = seq_block_ids_vals[offset : offset + block_count]

            # mirror get_args_data logic
            block_ids = req.cache_page_indices(block_count)
            expected = block_ids + [0] * (block_count - len(block_ids))

            assert results == expected

    def test_get_args(self, lsys, prefill_task: PrefillTask, page_pool):
        async def _test():
            batch_size = len(prefill_task.exec_requests)
            page_tables = page_pool.acquire_free_pages(batch_size)

            (args, req_count) = await prefill_task.get_args(
                page_tables=page_tables,
                batch_size=batch_size,
            )

            assert all(isinstance(arg, Allocation) for arg in args[:3])
            assert isinstance(args[-1], WrappedAllocation)
            assert req_count == len(prefill_task.exec_requests)

        lsys.run(_test())

    def test_get_results(
        self, lsys, prefill_task: PrefillTask, fiber, result_logits_none_indices
    ):
        async def _test():
            device0 = fiber.device(0)
            logits, indices = result_logits_none_indices
            await device0

            await prefill_task.get_result(
                logits=logits,
                indices=indices,
                req_count=len(prefill_task.exec_requests),
            )

            await device0
            for req in prefill_task.exec_requests:
                seq_len = len(req.input_token_ids) - 1
                results = req.result_logits.items.tolist()

                assert results == [_ for _ in range(seq_len)] + [0] * (16 - seq_len)

        lsys.run(_test())

    def test_get_results_w_indices(
        self, lsys, prefill_task: PrefillTask, fiber, result_logits_w_indices
    ):
        async def _test():
            device0 = fiber.device(0)
            batch_size = len(prefill_task.exec_requests)

            logits, indices = result_logits_w_indices
            await prefill_task.get_result(
                logits=logits,
                indices=indices,
                req_count=batch_size,
            )
            await device0

            # Verify get_result picked the exact [i, sl, :] vectors
            for i, req in enumerate(prefill_task.exec_requests):
                assert req.result_logits.items.tolist() == [i, i + 1, i + 2, i + 3]
                assert req.result_indices.items.tolist() == [
                    10 + i,
                    11 + i,
                    12 + i,
                    13 + i,
                ]

        lsys.run(_test())

    def test_post_process_logits(
        self,
        fiber,
        lsys,
        page_pool,
        prefill_task: PrefillTask,
        result_logits_none_indices,
    ):
        async def _test():
            batch_size = len(prefill_task.exec_requests)
            device0 = fiber.device(0)
            page_tables = page_pool.acquire_free_pages(batch_size)
            await device0

            args, _ = await prefill_task.get_args(
                page_tables=page_tables,
                batch_size=batch_size,
            )

            logits, _ = result_logits_none_indices
            vocab_size = logits.shape[-1]
            await prefill_task.post_process_logits(
                args=args,
                req_count=batch_size,
                result=(logits,),
                device0=device0,
            )

            # Verify that the logits were processed correctly
            for req in prefill_task.exec_requests:
                sl = len(req.input_token_ids) - 1
                expected = _pad_list(
                    [i for i in range(sl)],
                    vocab_size,
                )
                result = req.result_logits.items.tolist()
                assert result == expected

        lsys.run(_test())


class TestDecodeTask:
    def test_get_args_data(self, decode_task):
        block_count = max(
            len(req.allocation.pages) for req in decode_task.exec_requests
        )

        (
            token_vals,
            seq_len_vals,
            start_positions_vals,
            seq_block_ids_vals,
        ) = decode_task.get_args_data(
            exec_requests=decode_task.exec_requests,
            block_count=block_count,
        )

        for i, req in enumerate(decode_task.exec_requests):
            assert token_vals[i] == req.input_token_ids[-1]

        assert seq_len_vals == [
            req.start_position + 1 for req in decode_task.exec_requests
        ]
        assert start_positions_vals == [
            req.start_position for req in decode_task.exec_requests
        ]

        for i, req in enumerate(decode_task.exec_requests):
            offset = i * block_count
            results = seq_block_ids_vals[offset : offset + block_count]

            # mirror get_args_data logic
            block_ids = req.cache_page_indices(block_count)
            expected = _pad_list(block_ids, block_count)
            assert results == expected

    def test_get_args(self, lsys, decode_task: DecodeTask, page_pool):
        async def _test():
            batch_size = len(decode_task.exec_requests)
            page_tables = page_pool.acquire_free_pages(batch_size)

            (args, req_count) = await decode_task.get_args(
                page_tables=page_tables,
                batch_size=batch_size,
            )

            assert all(isinstance(arg, Allocation) for arg in args[:3])
            assert isinstance(args[-1], WrappedAllocation)
            assert req_count == len(decode_task.exec_requests)

        lsys.run(_test())

    def test_get_results(
        self, lsys, decode_task: DecodeTask, fiber, result_logits_none_indices_decode
    ):
        async def _test():
            device0 = fiber.device(0)
            logits, indices = result_logits_none_indices_decode
            vocab_size = logits.shape[-1]
            await device0

            await decode_task.get_result(
                logits=logits,
                indices=indices,
                req_count=len(decode_task.exec_requests),
            )

            for req in decode_task.exec_requests:
                results = req.result_logits.items.tolist()

                assert results == [_ for _ in range(vocab_size)]

                assert (
                    req.result_indices is None
                ), "Indices should be None for decode task"

        lsys.run(_test())

    def test_get_results_w_indices(
        self, lsys, decode_task: DecodeTask, fiber, result_logits_w_indices_decode
    ):
        async def _test():
            device0 = fiber.device(0)
            batch_size = len(decode_task.exec_requests)

            logits, indices = result_logits_w_indices_decode
            await device0

            await decode_task.get_result(
                logits=logits,
                indices=indices,
                req_count=batch_size,
            )

            # Verify get_result picked the exact [i, sl, :] vectors
            for i, req in enumerate(decode_task.exec_requests):
                assert req.result_logits.items.tolist() == [i, i + 1, i + 2, i + 3]
                assert req.result_indices.items.tolist() == [
                    10 + i,
                    11 + i,
                    12 + i,
                    13 + i,
                ]

        lsys.run(_test())


class TestLlmInvoker:
    def test_run_none_indices(
        self, lsys, llm_invoker: LlmInvoker, prefill_task, result_logits_none_indices
    ):
        async def _test():
            async def entrypoint(*args, fiber=None):
                return result_logits_none_indices

            llm_invoker.functions = {len(prefill_task.exec_requests): entrypoint}
            await llm_invoker.run()

            logits, _ = result_logits_none_indices
            vocab_size = logits.shape[-1]
            for req in prefill_task.exec_requests:
                seq_len = len(req.input_token_ids) - 1

                expected = _pad_list(
                    [_ for _ in range(seq_len)],
                    vocab_size,
                )
                results = req.result_logits.items.tolist()

                assert results == expected

        lsys.run(_test())
