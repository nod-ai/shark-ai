# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from shortfin_apps.llm.components.batching.modes.extend_attention import (
    ExtendAttentionScheduler,
)
from shortfin_apps.llm.components.invocation import LlmTaskInput


class FakeTaskInput:
    """Helper to create fake LlmTaskInput objects for testing."""

    def __init__(self, rid: str, token_count: int, instance_id: int = 0):
        self.rid = rid
        self.instance_id = instance_id
        self.input_tokens = tuple(range(token_count))
        self.seq_len = token_count
        self.block_count = 1
        self.page_ids = (0,)
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
    """Tests for ExtendAttentionScheduler."""

    def test_initialization(self):
        """Test that scheduler initializes correctly."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        assert scheduler.token_budget == 1024
        assert scheduler.block_seq_stride == 16
        assert len(scheduler._pending_by_length) == 0
        assert len(scheduler._pending_chunks) == 0

    def test_single_task_scheduling(self):
        """Test scheduling a single task."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        task = FakeTaskInput(rid="req1", token_count=100).to_task_input()

        scheduler.schedule_job(task)

        # Should be in ready queue
        assert len(scheduler._pending_by_length) == 1
        assert task.rid not in scheduler._pending_chunks

    def test_token_budget_batching(self):
        """Test that batching respects token budget, not request count."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Schedule tasks with varying token counts
        task1 = FakeTaskInput(rid="req1", token_count=500).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=400).to_task_input()
        task3 = FakeTaskInput(rid="req3", token_count=200).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)
        scheduler.schedule_job(task3)

        # Get batch - should fit task1 and task2 (900 tokens)
        batch = scheduler.get_next_batch()
        assert len(batch) == 2
        total_tokens = sum(len(t.input_tokens) for t in batch)
        assert total_tokens == 900
        assert total_tokens <= 1024

        # Get next batch - should have task3
        batch2 = scheduler.get_next_batch()
        assert len(batch2) == 1
        assert batch2[0].rid == "req3"

    def test_chunk_sequencing_first_chunk(self):
        """Test that first chunk of a request goes to ready queue."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Simulate first chunk
        first_chunk = FakeTaskInput(rid="req1", token_count=100).to_task_input()
        scheduler.schedule_job(first_chunk)

        # First chunk should be ready
        assert len(scheduler._pending_by_length) == 1
        assert "req1" not in scheduler._pending_chunks

    def test_chunk_sequencing_subsequent_chunks(self):
        """Test that subsequent chunks go to pending queue."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Schedule first chunk
        first_chunk = FakeTaskInput(rid="req1", token_count=100).to_task_input()
        scheduler.schedule_job(first_chunk)

        # Schedule second chunk (simulate by calling schedule_job again)
        second_chunk = FakeTaskInput(rid="req1", token_count=100).to_task_input()
        scheduler.schedule_job(second_chunk)

        # First chunk should be ready, second should be pending
        batch = scheduler.get_next_batch()
        assert len(batch) == 1
        assert batch[0] == first_chunk
        assert "req1" in scheduler._pending_chunks
        assert len(scheduler._pending_chunks["req1"]) == 1

    def test_handle_completed_promotes_next_chunk(self):
        """Test that completing a chunk promotes the next chunk to ready."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Schedule three chunks for same request
        chunk1 = FakeTaskInput(rid="req1", token_count=100).to_task_input()
        chunk2 = FakeTaskInput(rid="req1", token_count=100).to_task_input()
        chunk3 = FakeTaskInput(rid="req1", token_count=100).to_task_input()

        scheduler.schedule_job(chunk1)
        scheduler.schedule_job(chunk2)
        scheduler.schedule_job(chunk3)

        # Get first batch
        batch1 = scheduler.get_next_batch()
        assert len(batch1) == 1
        assert batch1[0] == chunk1

        # Complete first chunk - should promote second chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is False  # Still has chunks pending

        # Second chunk should now be ready
        batch2 = scheduler.get_next_batch()
        assert len(batch2) == 1
        # Note: We can't directly compare objects, but we can check the rid
        assert batch2[0].rid == "req1"

        # Complete second chunk - should promote third chunk
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is False

        # Third chunk should now be ready
        batch3 = scheduler.get_next_batch()
        assert len(batch3) == 1
        assert batch3[0].rid == "req1"

        # Complete third chunk - request should be fully complete
        is_complete = scheduler.handle_completed("req1")
        assert is_complete is True
        assert "req1" not in scheduler._pending_chunks

    def test_cross_request_batching(self):
        """Test that chunks from different requests can batch together."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Schedule first chunks from multiple requests
        task1 = FakeTaskInput(rid="req1", token_count=300).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=400).to_task_input()
        task3 = FakeTaskInput(rid="req3", token_count=200).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)
        scheduler.schedule_job(task3)

        # Should batch task1 and task2 together (700 tokens)
        batch = scheduler.get_next_batch()
        assert len(batch) == 2
        total_tokens = sum(len(t.input_tokens) for t in batch)
        assert total_tokens == 700

    def test_length_based_prioritization(self):
        """Test that scheduler prioritizes longer sequences."""
        scheduler = ExtendAttentionScheduler(token_budget=2000, block_seq_stride=16)

        # Schedule tasks with different lengths
        task_short = FakeTaskInput(rid="req_short", token_count=100).to_task_input()
        task_medium = FakeTaskInput(rid="req_medium", token_count=500).to_task_input()
        task_long = FakeTaskInput(rid="req_long", token_count=1000).to_task_input()

        # Schedule in random order
        scheduler.schedule_job(task_short)
        scheduler.schedule_job(task_long)
        scheduler.schedule_job(task_medium)

        # Should get longest first
        batch = scheduler.get_next_batch()
        assert batch[0].rid == "req_long"

    def test_empty_scheduler(self):
        """Test getting batch from empty scheduler."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        batch = scheduler.get_next_batch()
        assert len(batch) == 0

    def test_handle_completed_nonexistent_request(self):
        """Test completing a request that doesn't exist."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)
        # Should handle gracefully
        is_complete = scheduler.handle_completed("nonexistent")
        assert is_complete is True

    def test_token_budget_exact_fit(self):
        """Test batching when tokens exactly match budget."""
        scheduler = ExtendAttentionScheduler(token_budget=1000, block_seq_stride=16)

        task1 = FakeTaskInput(rid="req1", token_count=500).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=500).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)

        batch = scheduler.get_next_batch()
        assert len(batch) == 2
        total_tokens = sum(len(t.input_tokens) for t in batch)
        assert total_tokens == 1000

    def test_token_budget_overflow_prevention(self):
        """Test that scheduler doesn't exceed token budget."""
        scheduler = ExtendAttentionScheduler(token_budget=1000, block_seq_stride=16)

        task1 = FakeTaskInput(rid="req1", token_count=600).to_task_input()
        task2 = FakeTaskInput(rid="req2", token_count=500).to_task_input()

        scheduler.schedule_job(task1)
        scheduler.schedule_job(task2)

        # Should only get task1 (600 tokens), not both
        batch = scheduler.get_next_batch()
        assert len(batch) == 1
        assert batch[0].rid == "req1"

        # task2 should be in next batch
        batch2 = scheduler.get_next_batch()
        assert len(batch2) == 1
        assert batch2[0].rid == "req2"

    def test_multiple_chunks_different_requests(self):
        """Test handling multiple chunks from different requests concurrently."""
        scheduler = ExtendAttentionScheduler(token_budget=1024, block_seq_stride=16)

        # Request 1: 3 chunks
        r1_chunk1 = FakeTaskInput(rid="req1", token_count=200).to_task_input()
        r1_chunk2 = FakeTaskInput(rid="req1", token_count=200).to_task_input()
        r1_chunk3 = FakeTaskInput(rid="req1", token_count=200).to_task_input()

        # Request 2: 2 chunks
        r2_chunk1 = FakeTaskInput(rid="req2", token_count=300).to_task_input()
        r2_chunk2 = FakeTaskInput(rid="req2", token_count=300).to_task_input()

        # Schedule first chunks
        scheduler.schedule_job(r1_chunk1)
        scheduler.schedule_job(r2_chunk1)

        # Schedule subsequent chunks
        scheduler.schedule_job(r1_chunk2)
        scheduler.schedule_job(r1_chunk3)
        scheduler.schedule_job(r2_chunk2)

        # First batch: r1_chunk1 and r2_chunk1
        batch1 = scheduler.get_next_batch()
        assert len(batch1) == 2
        rids = {t.rid for t in batch1}
        assert rids == {"req1", "req2"}

        # Complete both
        scheduler.handle_completed("req1")
        scheduler.handle_completed("req2")

        # Second batch: r1_chunk2 and r2_chunk2
        batch2 = scheduler.get_next_batch()
        assert len(batch2) == 2
        rids = {t.rid for t in batch2}
        assert rids == {"req1", "req2"}

        # Complete both
        scheduler.handle_completed("req1")
        req2_complete = scheduler.handle_completed("req2")

        assert req2_complete is True  # req2 finished

        # Third batch: only r1_chunk3
        batch3 = scheduler.get_next_batch()
        assert len(batch3) == 1
        assert batch3[0].rid == "req1"

        req1_complete = scheduler.handle_completed("req1")
        assert req1_complete is True  # req1 finished


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
