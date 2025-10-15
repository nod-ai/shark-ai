# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import List
from dataclasses import dataclass
try:
    from sortedcontainers import SortedDict
except ImportError:
    # Fallback to regular dict if sortedcontainers is not available
    SortedDict = dict

import shortfin as sf
import shortfin.array as sfnp

from shortfin import Fiber

from ..batching_trait import BatchingTrait
from ..config import BatchConfig

from ...config_struct import ModelParams
from ...device_array_cache import DeviceArrayCache
from ...invocation import (
    PrefillTask,
    LlmInvocationProcess,
    LlmTask,
    LlmTaskInput,
)
from ...kvcache.base_attention_cache import BasePagedAttentionCache
from ...messages import InferencePhase, LlmInferenceExecRequest
from ...scheduler import AbstractScheduler

from .default import (
    LlmBatcherProcess,
    PrefillTaskResponder,
    DecodeBatcherProcess,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtendAttentionBatch:
    """Represents a batch of requests with varying prefill lengths for extend-attention."""
    task_inputs: List[LlmTaskInput]
    max_seq_len: int
    total_tokens: int

    def __init__(self, task_inputs: List[LlmTaskInput]):
        self.task_inputs = task_inputs
        self.max_seq_len = max(task.seq_len for task in task_inputs)
        self.total_tokens = sum(len(task.input_tokens) for task in task_inputs)


class ExtendAttentionScheduler(AbstractScheduler):
    """Scheduler that understands extend-attention batching constraints."""

    def __init__(self, *, ideal_batch_size: int, block_seq_stride: int):
        super().__init__(ideal_batch_size=ideal_batch_size)
        self.block_seq_stride = block_seq_stride
        # Use SortedDict to maintain tasks sorted by token count (descending order)
        # Key is negative token count for descending order, value is list of tasks
        self._pending_by_length: SortedDict = SortedDict()

    def schedule_job(self, task: LlmTaskInput):
        """Add a task to the scheduler, maintaining sorted order by token count."""
        token_count = len(task.input_tokens)
        # Use negative for descending order (largest first)
        key = -token_count

        if key not in self._pending_by_length:
            self._pending_by_length[key] = []
        self._pending_by_length[key].append(task)

    def should_execute(self, strobe) -> List[List[LlmTaskInput]]:
        """Determine which tasks should be executed now.

        With extend-attention, we can batch together prefill requests of different
        lengths more efficiently. The kernel will handle the varying lengths per page.
        """
        if not self._pending_by_length:
            return []

        # Collect all pending tasks from the sorted dict
        pending = []
        for tasks_list in self._pending_by_length.values():
            pending.extend(tasks_list)

        # Create batches using the already-sorted tasks
        batches = self._create_extend_attention_batches(pending)

        # Remove scheduled tasks from the sorted dict
        scheduled_tasks = set()
        for batch in batches:
            for task in batch:
                scheduled_tasks.add(task)

        # Rebuild the sorted dict with only unscheduled tasks
        new_pending = SortedDict()
        for key, tasks_list in self._pending_by_length.items():
            remaining = [t for t in tasks_list if t not in scheduled_tasks]
            if remaining:
                new_pending[key] = remaining

        self._pending_by_length = new_pending

        return batches

    def _create_extend_attention_batches(self, tasks: List[LlmTaskInput]) -> List[List[LlmTaskInput]]:
        """Create batches optimized for extend-attention.

        With extend-attention, we can batch tasks based on token budget only.
        No limit on number of requests per batch - only token budget matters.
        This allows for more flexible batching of variable-length sequences.

        Note: Tasks are already sorted in descending order by token count from SortedDict.
        """
        batches = []
        remaining = tasks.copy()

        while remaining:
            batch = []
            # Token budget is the only constraint - no limit on number of requests
            # This allows batching many decode requests (1 token each) with prefill chunks
            batch_token_budget = self.block_seq_stride * self._ideal_batch_size
            current_token_count = 0

            # Tasks are already sorted by length (descending) from SortedDict
            for task in remaining[:]:
                task_tokens = len(task.input_tokens)
                # For prefill: round up to page boundary
                # For decode: always 1 token (but this is handled by the task creation)
                task_pages = math.ceil(task_tokens / self.block_seq_stride)
                task_padded_tokens = task_pages * self.block_seq_stride

                # Only constraint: total tokens must fit in budget
                if current_token_count + task_padded_tokens <= batch_token_budget:
                    batch.append(task)
                    remaining.remove(task)
                    current_token_count += task_padded_tokens
                else:
                    # If this task doesn't fit, no smaller tasks will fit either
                    # (since tasks are sorted by length descending)
                    break

            if batch:
                batches.append(batch)

        return batches

    def handle_scheduler(self, msg) -> bool:
        # Handle scheduler messages
        return False

    def reserve_workload(self, *, batcher, count, rid):
        # Handle workload reservation
        pass

    def handle_completed(self, rid: str) -> bool:
        return True


class ExtendAttentionPrefillTask(PrefillTask):
    """Prefill task that supports extend-attention with varying sequence lengths."""

    def __init__(
        self,
        task_inputs: List[LlmTaskInput],
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
        has_prefill_position: bool,
        block_seq_stride: int,
    ):
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
            has_prefill_position=has_prefill_position,
        )
        self.block_seq_stride = block_seq_stride

    async def prepare_args(self, batch_size: int) -> List[sfnp.device_array]:
        """Prepare arguments for extend-attention prefill.

        With extend-attention, each request's tokens are divided into pages of
        block_seq_stride size. Each page can track its own history, allowing
        efficient batching of variable-length sequences.
        """
        task_inputs = self._task_inputs

        # For extend-attention, we need to prepare the batch with page-aligned tokens
        tokens = []
        seq_lens = []
        page_ids = []
        start_positions = []

        # Calculate the maximum blocks needed across all requests
        max_blocks = max(task.block_count for task in task_inputs)

        for task_input in task_inputs:
            # Each task's tokens are organized by pages
            task_tokens = list(task_input.input_tokens)

            # With extend-attention, we pad each sequence to page boundaries
            # This allows the kernel to handle per-page history tracking
            tokens.append(task_tokens)
            seq_lens.append(task_input.seq_len)
            page_ids.append(list(task_input.page_ids))
            if self._has_prefill_position:
                start_positions.append(task_input.start_position)

        # For extend-attention, calculate the maximum number of pages needed
        max_pages_needed = max(
            math.ceil(len(t) / self.block_seq_stride) for t in tokens
        )
        # Pad to page boundary - this is crucial for extend-attention
        max_seq_len = max_pages_needed * self.block_seq_stride

        logger.debug(
            f"ExtendAttention Prefill bs={batch_size}, "
            f"max_seq_len={max_seq_len}, max_pages={max_pages_needed}, "
            f"max_blocks={max_blocks}, tokens_per_page={self.block_seq_stride}"
        )

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Allocate buffers
        tokens_allocation = array_cache.allocate([batch_size, max_seq_len], int_dtype)
        seq_lens_allocation = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids_allocation = array_cache.allocate([batch_size, max_blocks], int_dtype)

        # Prepare data with padding
        from itertools import chain

        def _pad_list(data: List[int], target_length: int) -> List[int]:
            return data + [0] * max(0, target_length - len(data))

        tokens_data = list(
            chain.from_iterable(_pad_list(t, max_seq_len) for t in tokens)
        )

        seq_block_ids_data = list(
            chain.from_iterable(
                _pad_list(pages, target_length=max_blocks) for pages in page_ids
            )
        )

        buffers = [tokens_allocation]
        data = [tokens_data]
        defaults = [0]

        if self._has_prefill_position:
            start_positions_allocation = array_cache.allocate([batch_size], int_dtype)
            buffers.append(start_positions_allocation)
            data.append(start_positions)
            defaults.append(0)

        buffers.extend([seq_lens_allocation, seq_block_ids_allocation])
        data.extend([seq_lens, seq_block_ids_data])
        defaults.extend([1, 0])

        from ...buffers import create_argument_buffers
        from ...device_array_cache import WrappedAllocation

        args = create_argument_buffers(
            buffers=buffers,
            data=data,
            defaults=defaults,
        )

        for page_table in self._page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args


class ExtendAttentionPrefillBatcherProcess(LlmBatcherProcess):
    """Batcher process optimized for extend-attention prefill."""

    STROBE_SHORT_DELAY = 0.065
    STROBE_LONG_DELAY = 0.065

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        prefill_functions: dict[int, sf.ProgramFunction],
        program_isolation: str,
    ):
        # Use the extend-attention aware scheduler
        ideal_batch_size = max(model_params.prefill_batch_sizes)
        block_seq_stride = model_params.paged_kv_cache.block_seq_stride

        scheduler = ExtendAttentionScheduler(
            ideal_batch_size=ideal_batch_size,
            block_seq_stride=block_seq_stride
        )

        llm_task_responder = PrefillTaskResponder(scheduler=scheduler)

        super().__init__(
            name="extend_attention_prefill",
            fiber=fiber,
            page_cache=page_cache,
            model_params=model_params,
            functions=prefill_functions,
            ideal_batch_size=ideal_batch_size,
            program_isolation=program_isolation,
            scheduler=scheduler,
            llm_task_responder=llm_task_responder,
        )

    def make_task_inputs(
        self, exec_request: LlmInferenceExecRequest
    ) -> List[LlmTaskInput]:
        """Create variable-size chunks for extend-attention prefill.

        This method implements the variable chunking strategy described in the
        design document, where long sequences are divided into chunks that can
        be batched together efficiently.
        """
        # Configurable ideal chunk size (can be made configurable later)
        ideal_chunk_tokens = 128  # Default chunk size
        task_inputs = []

        total_tokens = len(exec_request.input_token_ids)

        # If the sequence is short enough, process it as a single chunk
        if total_tokens <= ideal_chunk_tokens:
            return [
                LlmTaskInput(
                    rid=exec_request.orig_instance_id,
                    instance_id=exec_request.instance_id,
                    block_count=exec_request.block_count,
                    seq_stride=self.page_seq_stride,
                    seq_len=total_tokens,
                    input_tokens=tuple(exec_request.input_token_ids),
                    page_ids=tuple(exec_request.page_ids),
                    start_position=exec_request.start_position,
                )
            ]

        # For longer sequences, create multiple chunks
        for chunk_start in range(0, total_tokens, ideal_chunk_tokens):
            chunk_end = min(chunk_start + ideal_chunk_tokens, total_tokens)
            chunk_tokens = exec_request.input_token_ids[chunk_start:chunk_end]

            task_inputs.append(LlmTaskInput(
                rid=exec_request.orig_instance_id,
                instance_id=exec_request.instance_id,
                block_count=exec_request.block_count,
                seq_stride=self.page_seq_stride,
                seq_len=len(chunk_tokens),              # Actual chunk length
                input_tokens=tuple(chunk_tokens),
                page_ids=tuple(exec_request.page_ids),  # All pages
                start_position=chunk_start,              # Absolute position
            ))

        return task_inputs

    def make_task(
        self,
        task_inputs: List[LlmTaskInput],
        page_cache: BasePagedAttentionCache,
    ) -> LlmTask:
        """Create an extend-attention aware prefill task."""
        return ExtendAttentionPrefillTask(
            task_inputs=task_inputs,
            array_cache=self.array_cache,
            page_tables=page_cache.page_pool.page_tables,
            has_prefill_position=self.model_params.has_prefill_position,
            block_seq_stride=self.page_seq_stride,
        )

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        task_inputs: list[LlmTaskInput],
    ) -> LlmInvocationProcess:
        """Create invoker for extend-attention prefill."""
        return LlmInvocationProcess(
            name="extend_attention_prefill_invocation",
            fiber=fiber,
            llm_task=self.make_task(task_inputs, page_cache),
            functions=self.functions,
            program_isolation=self.program_isolation,
            responder=self._llm_task_responder,
        )


class ExtendAttentionBatchingEngine(BatchingTrait):
    """Batching engine that uses extend-attention for improved prefill batching."""

    def __init__(
        self,
        prefill_lane: ExtendAttentionPrefillBatcherProcess,
        decode_lane: DecodeBatcherProcess,
    ):
        self.prefill_lane = prefill_lane
        self.decode_lane = decode_lane

    def submit(self, request: LlmInferenceExecRequest):
        if request.phase == InferencePhase.PREFILL:
            self.prefill_lane.submit(request)
        elif request.phase == InferencePhase.DECODE:
            self.decode_lane.submit(request)
        else:
            raise ValueError(
                "Requested unsupported batching lane: Supported only either prefill or decode."
            )

    def launch(self):
        self.prefill_lane.launch()
        self.decode_lane.launch()

    def shutdown(self):
        self.prefill_lane.shutdown()
        self.decode_lane.shutdown()

    def reserve_workload(self, rid: str, count: int):
        self.decode_lane.reserve_workload(rid=rid, count=count)

    def get_model_params(self) -> ModelParams:
        return self.prefill_lane.model_params

    @staticmethod
    def create(
        batch_cfg: BatchConfig,
        page_cache: BasePagedAttentionCache,
        prefill_fiber: sf.Fiber,
        decode_fiber: sf.Fiber | None = None,
    ):
        """Create an extend-attention batching engine."""
        assert decode_fiber is not None, "Decode fiber is required"

        # Check if the model was exported with extend-attention support
        if not batch_cfg.model_params.use_extend_attention:
            raise ValueError(
                "Model was not exported with extend-attention support. "
                "Please export the model with --use-extend-attention flag."
            )

        prefill_batcher = ExtendAttentionPrefillBatcherProcess(
            fiber=prefill_fiber,
            page_cache=page_cache,
            model_params=batch_cfg.model_params,
            prefill_functions=batch_cfg.prefill_functions,
            program_isolation=batch_cfg.prog_isolation,
        )

        decode_batcher = DecodeBatcherProcess(
            fiber=decode_fiber,
            page_cache=page_cache,
            model_params=batch_cfg.model_params,
            decode_functions=batch_cfg.decode_functions,
            program_isolation=batch_cfg.prog_isolation,
        )

        return ExtendAttentionBatchingEngine(
            prefill_lane=prefill_batcher,
            decode_lane=decode_batcher,
        )