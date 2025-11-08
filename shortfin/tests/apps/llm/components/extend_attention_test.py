# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import math
import pytest

import shortfin.array as sfnp

from typing import List
from unittest.mock import patch
from uuid import uuid4

from shortfin_apps.llm.components.scheduler import ExtendAttentionScheduler
from shortfin_apps.llm.components.invocation import ExtendAttentionPrefillTask
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.device_array_cache import (
    Allocation,
    WrappedAllocation,
)
from shortfin_apps.llm.components.invocation import LlmTaskInput
from shortfin_apps.llm.components.kvcache.attention_cache_abstract import CacheInfo
from shortfin_apps.llm.components.kvcache.page_pool import PageInfo
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


@pytest.fixture
def model_params():
    return ModelParams(
        max_seq_len=2048,
        transformer_block_count=32,
        attn_head_dim=128,
        prefill_batch_sizes=[1, 2, 4],
        decode_batch_sizes=[1, 2, 4],
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=16,
            attention_head_count_kv=8,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture(scope="function")
def extend_attention_exec_requests(cache_ref_count, page_pool):
    """Create exec requests with varying token lengths for extend attention testing."""
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        exec_reqs = []
        token_lengths = [64, 128, 256, 512]  # Different lengths to test batching

        page_offset = 0
        for idx, token_len in enumerate(token_lengths):
            input_tokens = [i + idx * 1000 for i in range(token_len)]
            exec_req = LlmInferenceExecRequest(
                phase=InferencePhase.PREFILL,
                input_token_ids=input_tokens,
                rid=str(uuid4()),
            )
            exec_reqs.append(exec_req)

            # Allocate pages for this request
            exec_req._cache = cache_ref_count
            pages = [
                PageInfo(index=page_offset + i, pool=page_pool)
                for i in range(math.ceil(len(input_tokens) / 16))
            ]
            exec_req.allocated_cache_info = CacheInfo(
                num_tokens=len(exec_req.input_token_ids),
                tokens=exec_req.input_token_ids,
                pages=pages,
                pool=page_pool,
                last_cached_node=None,
            )
            exec_req.page_ids = [page.index for page in pages]
            page_offset += len(pages)

        yield exec_reqs


def _get_extend_attention_task_inputs(
    exec_requests: List[LlmInferenceExecRequest],
) -> List[LlmTaskInput]:
    """Convert exec requests to task inputs for extend attention."""
    task_inputs = []
    for req in exec_requests:
        task_inputs.append(
            LlmTaskInput(
                rid=req.orig_instance_id,
                instance_id=req.instance_id,
                block_count=req.block_count,
                input_tokens=tuple(req.input_token_ids),
                seq_len=len(req.input_token_ids),
                page_ids=tuple(req.page_ids),
                start_position=0,
            )
        )
    return task_inputs


@pytest.fixture(scope="function")
def extend_attention_prefill_task(
    extend_attention_exec_requests, device_array_cache, page_pool
) -> ExtendAttentionPrefillTask:
    """Fixture to create an ExtendAttentionPrefillTask."""
    page_tables = page_pool.acquire_free_pages(len(extend_attention_exec_requests))
    task_inputs = _get_extend_attention_task_inputs(extend_attention_exec_requests)
    return ExtendAttentionPrefillTask(
        task_inputs=task_inputs,
        array_cache=device_array_cache,
        page_tables=page_tables,
        has_prefill_position=False,
        block_seq_stride=16,
    )


@pytest.fixture(scope="function")
def extend_attention_prefill_task_w_start_pos(
    extend_attention_exec_requests, device_array_cache, page_pool
) -> ExtendAttentionPrefillTask:
    """Fixture to create an ExtendAttentionPrefillTask with start positions."""
    page_tables = page_pool.acquire_free_pages(len(extend_attention_exec_requests))
    task_inputs = _get_extend_attention_task_inputs(extend_attention_exec_requests)

    # Set different start positions to test offset prefill
    for i, task_input in enumerate(task_inputs):
        task_inputs[i] = LlmTaskInput(
            rid=task_input.rid,
            instance_id=task_input.instance_id,
            block_count=task_input.block_count,
            input_tokens=task_input.input_tokens,
            seq_len=task_input.seq_len,
            page_ids=task_input.page_ids,
            start_position=i * 16,  # Different start positions
        )

    return ExtendAttentionPrefillTask(
        task_inputs=task_inputs,
        array_cache=device_array_cache,
        page_tables=page_tables,
        has_prefill_position=True,
        block_seq_stride=16,
    )


class TestExtendAttentionScheduler:
    """Tests for ExtendAttentionScheduler - dynamic chunking logic."""

    def test_initialization(self):
        """Test scheduler initializes with correct parameters."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        assert scheduler._token_budget == 1024
        assert scheduler._block_seq_stride == 16
        assert len(scheduler._active_requests) == 0
        assert len(scheduler._request_positions) == 0
        assert len(scheduler._last_chunk_sizes) == 0

    def test_schedule_job_new_request(self):
        """Test scheduling a new job."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=10,
            seq_len=160,
            input_tokens=tuple(range(160)),
            page_ids=tuple(range(10)),
            start_position=0,
        )

        scheduler.schedule_job(task)
        assert "req1" in scheduler._active_requests
        assert scheduler._request_positions["req1"] == 0

    def test_schedule_job_with_start_position(self):
        """Test scheduling a job with non-zero start position (trie prefix matching)."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=10,
            seq_len=160,
            input_tokens=tuple(range(160)),
            page_ids=tuple(range(10)),
            start_position=32,
        )

        scheduler.schedule_job(task)
        assert "req1" in scheduler._active_requests
        assert scheduler._request_positions["req1"] == 32

    def test_schedule_duplicate_job(self, caplog):
        """Test scheduling a duplicate job logs warning."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=10,
            seq_len=160,
            input_tokens=tuple(range(160)),
            page_ids=tuple(range(10)),
            start_position=0,
        )

        scheduler.schedule_job(task)
        with caplog.at_level(logging.WARNING):
            scheduler.schedule_job(task)

        assert "already scheduled" in caplog.text

    def test_dynamic_chunk_calculation_single_request(self):
        """Test dynamic chunk size with single request gets full budget."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=128,
            seq_len=2048,
            input_tokens=tuple(range(2048)),
            page_ids=tuple(range(128)),
            start_position=0,
        )

        scheduler.schedule_job(task)
        batches = scheduler.should_execute(strobe=0)

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert len(batches[0][0].input_tokens) == 1024  # Full budget

    def test_dynamic_chunk_calculation_two_requests(self):
        """Test dynamic chunk size with two requests splits budget."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        for i in range(2):
            task = LlmTaskInput(
                rid=f"req{i}",
                instance_id=i,
                block_count=128,
                seq_len=2048,
                input_tokens=tuple(range(2048)),
                page_ids=tuple(range(128)),
                start_position=0,
            )
            scheduler.schedule_job(task)

        batches = scheduler.should_execute(strobe=0)

        assert len(batches) == 1
        assert len(batches[0]) == 2
        # Each request gets 512 tokens (1024 / 2, page-aligned)
        assert all(len(chunk.input_tokens) == 512 for chunk in batches[0])

    def test_page_alignment(self):
        """Test that chunk sizes are page-aligned."""
        scheduler = ExtendAttentionScheduler(token_budget=1000, block_seq_stride=16)

        for i in range(3):
            task = LlmTaskInput(
                rid=f"req{i}",
                instance_id=i,
                block_count=128,
                seq_len=2048,
                input_tokens=tuple(range(2048)),
                page_ids=tuple(range(128)),
                start_position=0,
            )
            scheduler.schedule_job(task)

        batches = scheduler.should_execute(strobe=0)

        # 1000 / 3 = 333.33 -> 320 (20 pages * 16)
        for chunk in batches[0]:
            assert len(chunk.input_tokens) % 16 == 0

    def test_handle_completed_advances_position(self):
        """Test that handle_completed advances position correctly."""
        scheduler = ExtendAttentionScheduler(token_budget=512, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=64,
            seq_len=1024,
            input_tokens=tuple(range(1024)),
            page_ids=tuple(range(64)),
            start_position=0,
        )

        scheduler.schedule_job(task)

        # First execution
        batches = scheduler.should_execute(strobe=0)
        assert batches[0][0].start_position == 0
        assert len(batches[0][0].input_tokens) == 512

        # Complete first chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is False
        assert scheduler._request_positions["req1"] == 512

        # Second execution
        batches = scheduler.should_execute(strobe=0)
        assert batches[0][0].start_position == 512
        assert len(batches[0][0].input_tokens) == 512

        # Complete second chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is True
        assert "req1" not in scheduler._active_requests

    def test_handle_completed_nonexistent_request(self):
        """Test handle_completed asserts on non-existent request."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        with pytest.raises(AssertionError):
            scheduler.handle_completed("nonexistent")

    def test_cumulative_metadata(self):
        """Test that seq_len and block_count are cumulative across chunks."""
        scheduler = ExtendAttentionScheduler(token_budget=512, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=64,
            seq_len=1024,
            input_tokens=tuple(range(1024)),
            page_ids=tuple(range(64)),
            start_position=0,
        )

        scheduler.schedule_job(task)

        # First chunk
        batches = scheduler.should_execute(strobe=0)
        chunk1 = batches[0][0]
        assert chunk1.seq_len == 512
        assert chunk1.block_count == 32  # ceil(512 / 16)

        scheduler.handle_completed("req1")

        # Second chunk
        batches = scheduler.should_execute(strobe=0)
        chunk2 = batches[0][0]
        assert chunk2.seq_len == 1024  # Cumulative
        assert chunk2.block_count == 64  # ceil(1024 / 16)

    def test_request_shorter_than_chunk(self):
        """Test request with fewer tokens than allocated chunk size."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=19,
            seq_len=300,
            input_tokens=tuple(range(300)),
            page_ids=tuple(range(19)),
            start_position=0,
        )
        scheduler.schedule_job(task)

        # Should get all 300 tokens (less than budget)
        batches = scheduler.should_execute(strobe=0)
        assert len(batches[0][0].input_tokens) == 300

        # Complete - should be done
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is True

    def test_dynamic_chunk_size_adjustment(self):
        """Test that chunk size adjusts as requests complete."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        # req1 is long, req2 is short (will complete in first batch)
        task1 = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=125,
            seq_len=2000,
            input_tokens=tuple(range(2000)),
            page_ids=tuple(range(125)),
            start_position=0,
        )
        task2 = LlmTaskInput(
            rid="req2",
            instance_id=1,
            block_count=25,
            seq_len=400,
            input_tokens=tuple(range(400)),
            page_ids=tuple(range(25)),
            start_position=0,
        )

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)

        # First batch: 512 tokens each (budget / 2, page-aligned)
        batches1 = scheduler.should_execute(strobe=0)
        assert len(batches1[0]) == 2
        # req1 gets 512, req2 gets all 400 (less than allocated 512)
        req1_chunk = [c for c in batches1[0] if c.rid == "req1"][0]
        req2_chunk = [c for c in batches1[0] if c.rid == "req2"][0]
        assert len(req1_chunk.input_tokens) == 512
        assert len(req2_chunk.input_tokens) == 400

        # Complete both chunks - req2 is done, req1 has more
        is_complete_req1 = scheduler.handle_completed("req1")
        is_complete_req2 = scheduler.handle_completed("req2")
        assert is_complete_req1 is False  # More tokens remain
        assert is_complete_req2 is True  # Request complete

        # Second batch: only req1 active - should get full budget (1024 tokens)
        batches2 = scheduler.should_execute(strobe=0)
        assert len(batches2[0]) == 1
        assert batches2[0][0].rid == "req1"
        assert len(batches2[0][0].input_tokens) == 1024
        assert batches2[0][0].start_position == 512

    def test_empty_scheduler(self):
        """Test getting batch from empty scheduler."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        batches = scheduler.should_execute(strobe=0)
        assert len(batches) == 0

    def test_page_ids_grow_with_chunks(self):
        """Test that page_ids include all blocks up to current position."""
        scheduler = ExtendAttentionScheduler(token_budget=256, block_seq_stride=16)
        task = LlmTaskInput(
            rid="req1",
            instance_id=0,
            block_count=63,
            seq_len=1000,
            input_tokens=tuple(range(1000)),
            page_ids=tuple(range(63)),
            start_position=0,
        )
        scheduler.schedule_job(task)

        # First chunk
        batches1 = scheduler.should_execute(strobe=0)
        chunk1 = batches1[0][0]
        assert len(chunk1.page_ids) == chunk1.block_count

        # Complete and get second chunk
        scheduler.handle_completed("req1")
        batches2 = scheduler.should_execute(strobe=0)
        chunk2 = batches2[0][0]
        assert len(chunk2.page_ids) == chunk2.block_count
        # Second chunk should have more pages than first
        assert chunk2.block_count > chunk1.block_count


class TestExtendAttentionPrefillTask:
    """Tests for ExtendAttentionPrefillTask - argument preparation."""

    def test_initialization(
        self, extend_attention_prefill_task: ExtendAttentionPrefillTask
    ):
        """Test task initializes correctly."""
        assert extend_attention_prefill_task._seq_stride == 16
        assert extend_attention_prefill_task.req_count == 4

    def test_prepare_args_structure(
        self, lsys, extend_attention_prefill_task: ExtendAttentionPrefillTask
    ):
        """Test that prepare_args returns correct structure."""

        async def _test():
            args = await extend_attention_prefill_task.prepare_args(batch_size=4)

            # Should have tokens, seq_lens, seq_block_ids, plus page tables
            assert len(args) >= 3
            assert all(isinstance(arg, Allocation) for arg in args[:3])
            assert all(isinstance(arg, WrappedAllocation) for arg in args[3:])

        lsys.run(_test())

    def test_prepare_args_shapes(
        self,
        lsys,
        extend_attention_prefill_task: ExtendAttentionPrefillTask,
        extend_attention_exec_requests,
    ):
        """Test that prepared arguments have correct shapes."""

        async def _test():
            batch_size = len(extend_attention_exec_requests)
            args = await extend_attention_prefill_task.prepare_args(
                batch_size=batch_size
            )

            # Tokens should be [batch_size, max_seq_len]
            assert args[0].shape[0] == batch_size
            # Should be page-aligned
            assert args[0].shape[1] % 16 == 0

            # Seq lens should be [batch_size]
            assert args[1].shape[0] == batch_size

            # Seq block ids should be [batch_size, max_blocks]
            assert args[2].shape[0] == batch_size

        lsys.run(_test())

    def test_prepare_args_with_start_position(
        self,
        lsys,
        extend_attention_prefill_task_w_start_pos: ExtendAttentionPrefillTask,
    ):
        """Test prepare_args with start positions included."""

        async def _test():
            args = await extend_attention_prefill_task_w_start_pos.prepare_args(
                batch_size=4
            )

            # Should have tokens, start_positions, seq_lens, seq_block_ids
            assert len(args) >= 4

            # Start positions should be [batch_size]
            assert args[1].shape[0] == 4

            # Verify start positions match what we set
            start_positions = args[1].host.items.tolist()
            assert start_positions == [0, 16, 32, 48]

        lsys.run(_test())

    def test_padding_to_page_boundaries(
        self,
        lsys,
        extend_attention_prefill_task: ExtendAttentionPrefillTask,
    ):
        """Test that sequences are padded to page boundaries."""

        async def _test():
            args = await extend_attention_prefill_task.prepare_args(batch_size=4)

            tokens_alloc = args[0]
            max_seq_len = tokens_alloc.shape[1]

            # Max seq len should be divisible by block_seq_stride
            assert max_seq_len % 16 == 0

            # Should accommodate the longest sequence (512 tokens)
            # which needs 32 pages, so 32 * 16 = 512
            assert max_seq_len >= 512

        lsys.run(_test())

    def test_varying_sequence_lengths(
        self,
        lsys,
        extend_attention_prefill_task: ExtendAttentionPrefillTask,
        extend_attention_exec_requests,
    ):
        """Test handling of requests with varying sequence lengths."""

        async def _test():
            args = await extend_attention_prefill_task.prepare_args(batch_size=4)

            seq_lens_alloc = args[1]
            seq_lens = seq_lens_alloc.host.items.tolist()

            # Should match the input token lengths
            expected_lens = [
                len(req.input_token_ids) for req in extend_attention_exec_requests
            ]
            assert seq_lens == expected_lens

        lsys.run(_test())

    def test_max_blocks_calculation(
        self,
        lsys,
        extend_attention_prefill_task: ExtendAttentionPrefillTask,
        extend_attention_exec_requests,
    ):
        """Test that max_blocks is calculated correctly."""

        async def _test():
            args = await extend_attention_prefill_task.prepare_args(batch_size=4)

            seq_block_ids_alloc = args[2]
            max_blocks = seq_block_ids_alloc.shape[1]

            # Should match the maximum block_count across all requests
            expected_max_blocks = max(
                req.block_count for req in extend_attention_exec_requests
            )
            assert max_blocks == expected_max_blocks

        lsys.run(_test())

    def test_offset_prefill_padding(
        self,
        lsys,
        device_array_cache,
        page_pool,
    ):
        """Test that seq_block_ids are padded correctly for offset prefill.

        This tests the edge case mentioned in PR review:
        Request A: 96 tokens, start_position=0, 6 pages
        Request B: 64 tokens, start_position=96, 5 pages

        max_blocks should be 6 (max_block_start) + 4 (write_block_span) = 10
        """

        async def _test():
            # Request A: 96 tokens from position 0
            task1 = LlmTaskInput(
                rid="reqA",
                instance_id="0",
                block_count=6,  # 96 / 16 = 6
                seq_len=96,
                input_tokens=tuple(range(96)),
                page_ids=tuple(range(6)),
                start_position=0,
            )

            # Request B: 64 tokens from position 96 (offset prefill)
            task2 = LlmTaskInput(
                rid="reqB",
                instance_id="1",
                block_count=10,  # (96 + 64) / 16 = 10 total
                seq_len=160,  # 96 + 64
                input_tokens=tuple(range(64)),
                page_ids=tuple(range(10)),  # All pages including history
                start_position=96,
            )

            page_tables = page_pool.acquire_free_pages(2)
            task = ExtendAttentionPrefillTask(
                task_inputs=[task1, task2],
                array_cache=device_array_cache,
                page_tables=page_tables,
                has_prefill_position=True,
                block_seq_stride=16,
            )

            args = await task.prepare_args(batch_size=2)

            # seq_block_ids should have shape [2, max_blocks]
            # max_blocks = max_block_start (6) + write_block_span (6) = 12
            # write_block_span = max_seq_len / 16 = 96 / 16 = 6
            seq_block_ids_alloc = args[3]
            assert seq_block_ids_alloc.shape[0] == 2  # batch size
            # max_blocks should accommodate both the max start position and the write span
            assert seq_block_ids_alloc.shape[1] >= 10  # At least 10 blocks for reqB

        lsys.run(_test())


class TestExtendAttentionIntegration:
    """Integration tests combining scheduler and task."""

    def test_full_prefill_workflow(
        self, lsys, device_array_cache, page_pool, cache_ref_count
    ):
        """Test complete workflow: schedule -> execute -> complete."""

        async def _test():
            scheduler = ExtendAttentionScheduler(token_budget=256, block_seq_stride=16)

            # Create two requests
            request_ids = []
            for i in range(2):
                input_tokens = [j + i * 1000 for j in range(512)]
                req = LlmInferenceExecRequest(
                    phase=InferencePhase.PREFILL,
                    input_token_ids=input_tokens,
                    rid=f"req{i}",
                )
                req._cache = cache_ref_count
                pages = [PageInfo(index=j + i * 32, pool=page_pool) for j in range(32)]
                req.allocated_cache_info = CacheInfo(
                    num_tokens=len(input_tokens),
                    tokens=input_tokens,
                    pages=pages,
                    pool=page_pool,
                    last_cached_node=None,
                )
                req.page_ids = [page.index for page in pages]

                # Schedule job using the actual request ID
                task_input = LlmTaskInput(
                    rid=req.orig_instance_id,
                    instance_id=req.instance_id,
                    block_count=req.block_count,
                    input_tokens=tuple(req.input_token_ids),
                    seq_len=len(req.input_token_ids),
                    page_ids=tuple(req.page_ids),
                    start_position=0,
                )
                scheduler.schedule_job(task_input)
                request_ids.append(req.orig_instance_id)

            # First batch - both requests get 128 tokens (256 / 2, page-aligned)
            batches = scheduler.should_execute(strobe=0)
            assert len(batches[0]) == 2
            assert all(len(chunk.input_tokens) == 128 for chunk in batches[0])

            # Create task from batch
            page_tables = page_pool.acquire_free_pages(2)
            task = ExtendAttentionPrefillTask(
                task_inputs=batches[0],
                array_cache=device_array_cache,
                page_tables=page_tables,
                has_prefill_position=False,
                block_seq_stride=16,
            )

            # Prepare arguments
            args = await task.prepare_args(batch_size=2)
            assert len(args) >= 3

            # Verify tokens allocation
            assert args[0].shape[0] == 2

            # Complete both chunks using actual request IDs
            scheduler.handle_completed(request_ids[0])
            scheduler.handle_completed(request_ids[1])

            # Second batch - both should continue from position 128
            batches = scheduler.should_execute(strobe=0)
            assert len(batches[0]) == 2
            assert all(chunk.start_position == 128 for chunk in batches[0])

        lsys.run(_test())
