# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import math
from shortfin_apps.llm.components.batching.modes.extend_attention import (
    ExtendAttentionScheduler,
)
from shortfin_apps.llm.components.invocation import LlmTaskInput


class FakeTaskInput:
    """Helper to create fake LlmTaskInput objects for testing."""

    def __init__(
        self,
        rid: str,
        token_count: int,
        instance_id: int = 0,
        block_seq_stride: int = 16,
    ):
        self.rid = rid
        self.instance_id = instance_id
        self.input_tokens = tuple(range(token_count))
        self.seq_len = token_count
        # Calculate block_count based on token_count
        self.block_count = math.ceil(token_count / block_seq_stride)
        # Generate page_ids for all blocks
        self.page_ids = tuple(range(self.block_count))
        self.start_position = 0

    def to_task_input(self) -> LlmTaskInput:
        return LlmTaskInput(
            rid=self.rid,
            instance_id=self.instance_id,
            block_count=self.block_count,
            seq_len=self.seq_len,
            input_tokens=self.input_tokens,
            page_ids=self.page_ids,
            start_position=self.start_position,
        )


class TestExtendAttentionScheduler:
    """Tests for ExtendAttentionScheduler with dynamic chunking."""

    def test_initialization(self):
        """Test that scheduler initializes correctly."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        assert scheduler.token_budget == 1024
        assert scheduler.block_seq_stride == 16
        assert len(scheduler._active_requests) == 0
        assert len(scheduler._request_positions) == 0

    def test_single_request_full_budget(self):
        """Test single request gets full token budget."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        # Request with 2000 tokens
        task = FakeTaskInput(rid="req1", token_count=2000).to_task_input()
        scheduler.schedule_job(task)

        # Should get 1024 tokens (full budget, page-aligned)
        batches = scheduler.should_execute(strobe=0)
        assert len(batches) == 1
        batch = batches[0]
        assert len(batch) == 1
        assert len(batch[0].input_tokens) == 1024
        assert batch[0].rid == "req1"
        assert batch[0].start_position == 0

    def test_two_requests_split_budget(self):
        """Test two requests split token budget equally."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task1 = FakeTaskInput(rid="req1", token_count=1000).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=1000).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)

        # Each should get 512 tokens (1024 / 2, page-aligned)
        batches = scheduler.should_execute(strobe=0)
        assert len(batches) == 1
        batch = batches[0]
        assert len(batch) == 2

        for chunk in batch:
            assert len(chunk.input_tokens) == 512
            assert chunk.start_position == 0

    def test_three_requests_split_budget(self):
        """Test three requests split budget with page alignment."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task1 = FakeTaskInput(rid="req1", token_count=1000).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=1000).to_task_input()
        task3 = FakeTaskInput(rid="req3", token_count=1000).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)
        scheduler.schedule_job(task3)

        # 1024 / 3 = 341.3 → 336 tokens (21 pages * 16)
        batches = scheduler.should_execute(strobe=0)
        assert len(batches) == 1
        batch = batches[0]
        assert len(batch) == 3

        for chunk in batch:
            assert len(chunk.input_tokens) == 336
            assert chunk.start_position == 0

    def test_handle_completed_advances_position(self):
        """Test that completing a chunk advances position for next chunk."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = FakeTaskInput(rid="req1", token_count=2000).to_task_input()
        scheduler.schedule_job(task)

        # First batch: 1024 tokens from position 0
        batches1 = scheduler.should_execute(strobe=0)
        assert batches1[0][0].start_position == 0
        assert len(batches1[0][0].input_tokens) == 1024

        # Complete first chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is False  # More tokens remain

        # Second batch: 976 remaining tokens from position 1024
        batches2 = scheduler.should_execute(strobe=0)
        assert batches2[0][0].start_position == 1024
        assert len(batches2[0][0].input_tokens) == 976

        # Complete second chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is True  # Request complete

    def test_request_shorter_than_chunk(self):
        """Test request with fewer tokens than allocated chunk size."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = FakeTaskInput(rid="req1", token_count=300).to_task_input()
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
        task1 = FakeTaskInput(rid="req1", token_count=2000).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=400).to_task_input()

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

    def test_handle_completed_nonexistent_request(self):
        """Test completing a request that doesn't exist."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        is_complete = scheduler.handle_completed("nonexistent")
        assert is_complete is True

    def test_page_alignment(self):
        """Test that all chunks are page-aligned."""
        scheduler = ExtendAttentionScheduler(token_budget=1000, block_seq_stride=16)
        task1 = FakeTaskInput(rid="req1", token_count=5000).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=5000).to_task_input()
        task3 = FakeTaskInput(rid="req3", token_count=5000).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)
        scheduler.schedule_job(task3)

        batches = scheduler.should_execute(strobe=0)
        for chunk in batches[0]:
            # All chunks should be divisible by block_seq_stride
            assert len(chunk.input_tokens) % 16 == 0

    def test_cumulative_seq_len_and_block_count(self):
        """Test that seq_len and block_count are cumulative."""
        scheduler = ExtendAttentionScheduler(token_budget=512, block_seq_stride=16)
        task = FakeTaskInput(rid="req1", token_count=1500).to_task_input()
        scheduler.schedule_job(task)

        # First chunk: 512 tokens
        batches1 = scheduler.should_execute(strobe=0)
        chunk1 = batches1[0][0]
        assert chunk1.seq_len == 512
        assert chunk1.block_count == 32  # ceil(512 / 16)
        assert len(chunk1.input_tokens) == 512

        # Complete first chunk
        scheduler.handle_completed("req1")

        # Second chunk: next 512 tokens (cumulative 1024)
        batches2 = scheduler.should_execute(strobe=0)
        chunk2 = batches2[0][0]
        assert chunk2.seq_len == 1024  # Cumulative
        assert chunk2.block_count == 64  # ceil(1024 / 16)
        assert len(chunk2.input_tokens) == 512
        assert chunk2.start_position == 512

        # Complete second chunk
        scheduler.handle_completed("req1")

        # Third chunk: remaining 476 tokens (cumulative 1500)
        batches3 = scheduler.should_execute(strobe=0)
        chunk3 = batches3[0][0]
        assert chunk3.seq_len == 1500  # Cumulative
        assert chunk3.block_count == 94  # ceil(1500 / 16)
        assert len(chunk3.input_tokens) == 476
        assert chunk3.start_position == 1024

    def test_page_ids_grow_with_chunks(self):
        """Test that page_ids include all blocks up to current position."""
        scheduler = ExtendAttentionScheduler(token_budget=256, block_seq_stride=16)
        task = FakeTaskInput(
            rid="req1", token_count=1000, block_seq_stride=16
        ).to_task_input()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
