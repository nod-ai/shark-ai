# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import Dict, List
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

    def __init__(self, *, token_budget: int, block_seq_stride: int):
        # Pass dummy ideal_batch_size to parent - not used in extend attention
        super().__init__(ideal_batch_size=1)
        self.block_seq_stride = block_seq_stride
        self.token_budget = token_budget
        # Track active requests (full task inputs with all tokens)
        self._active_requests: Dict[str, LlmTaskInput] = {}
        # Track current position (token offset) for each request
        self._request_positions: Dict[str, int] = {}

    def schedule_job(self, task: LlmTaskInput):
        """Add a request to the scheduler.

        The task contains all tokens for the request. We'll dynamically chunk it
        at scheduling time based on the number of active requests.
        """
        rid = task.rid
        if rid not in self._active_requests:
            # New request - store it and initialize position to 0
            self._active_requests[rid] = task
            self._request_positions[rid] = 0

    def should_execute(self, strobe) -> List[List[LlmTaskInput]]:
        """Determine which tasks should be executed now.

        Dynamically chunks active requests based on the number of requests and token budget.
        Each request gets a page-aligned chunk that fits within the budget.
        """
        if not self._active_requests:
            return []

        # Calculate dynamic chunk size based on active requests
        num_active = len(self._active_requests)
        tokens_per_request = self.token_budget // num_active
        # Align to page boundaries
        chunk_size = (tokens_per_request // self.block_seq_stride) * self.block_seq_stride

        if chunk_size == 0:
            # Too many requests for the budget - shouldn't happen but handle gracefully
            chunk_size = self.block_seq_stride

        # Create chunks for this batch
        batch = []
        for rid, full_task in self._active_requests.items():
            position = self._request_positions[rid]
            all_tokens = full_task.input_tokens

            # Determine how many tokens to take
            remaining_tokens = len(all_tokens) - position
            tokens_to_take = min(chunk_size, remaining_tokens)

            if tokens_to_take > 0:
                # Create a chunk from current position
                chunk_tokens = all_tokens[position:position + tokens_to_take]

                # Calculate cumulative seq_len and block_count
                cumulative_seq_len = position + len(chunk_tokens)
                chunk_block_count = math.ceil(cumulative_seq_len / self.block_seq_stride)

                # Get page_ids up to the current block count
                chunk_page_ids = full_task.page_ids[:chunk_block_count]

                # Create the chunk task input
                chunk_task = LlmTaskInput(
                    rid=rid,
                    instance_id=full_task.instance_id,
                    block_count=chunk_block_count,
                    seq_len=cumulative_seq_len,
                    input_tokens=chunk_tokens,
                    page_ids=chunk_page_ids,
                    start_position=position,
                )
                batch.append(chunk_task)

        return [batch] if batch else []


    def handle_scheduler(self, msg) -> bool:
        # Handle scheduler messages
        return False

    def reserve_workload(self, *, batcher, count, rid):
        # Handle workload reservation
        pass

    def handle_completed(self, rid: str) -> bool:
        """Handle completion of a chunk.

        Updates the position for this request and determines if more tokens remain.

        Returns True if the request is fully complete (no more tokens).
        Returns False if there are more tokens to process.
        """
        if rid not in self._active_requests:
            # Request not found (shouldn't happen, but handle gracefully)
            return True

        full_task = self._active_requests[rid]
        current_position = self._request_positions[rid]

        # Calculate how many tokens were in the last chunk we executed
        # We need to figure out what chunk size was used
        num_active = len(self._active_requests)
        tokens_per_request = self.token_budget // num_active
        chunk_size = (tokens_per_request // self.block_seq_stride) * self.block_seq_stride
        if chunk_size == 0:
            chunk_size = self.block_seq_stride

        # Advance position by the chunk size (or remaining tokens, whichever is smaller)
        remaining_tokens = len(full_task.input_tokens) - current_position
        tokens_processed = min(chunk_size, remaining_tokens)
        new_position = current_position + tokens_processed

        # Update position
        self._request_positions[rid] = new_position

        # Check if we've processed all tokens
        if new_position >= len(full_task.input_tokens):
            # Request complete - remove from active requests
            del self._active_requests[rid]
            del self._request_positions[rid]
            return True  # Request fully complete
        else:
            # More tokens to process
            return False


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
            seq_stride=block_seq_stride,
            has_prefill_position=has_prefill_position,
            chunk_block_size=None,
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
        # Pad to page boundary
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
        seq_block_ids_allocation = array_cache.allocate(
            [batch_size, max_blocks], int_dtype
        )

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
        token_budget: int,
    ):
        # Use the extend-attention aware scheduler
        block_seq_stride = model_params.paged_kv_cache.block_seq_stride

        scheduler = ExtendAttentionScheduler(
            token_budget=token_budget, block_seq_stride=block_seq_stride
        )

        llm_task_responder = PrefillTaskResponder(scheduler=scheduler)

        # ideal_batch_size - not really important. we can set it to
        #  maximum number of requests that can be batched together.
        ideal_batch_size = token_budget // block_seq_stride

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
        """Create a single task input containing all tokens.

        The scheduler will dynamically chunk this request at scheduling time based
        on the number of active requests and the token budget.
        """
        total_tokens = len(exec_request.input_token_ids)

        # Return a single task with ALL tokens
        # The scheduler will chunk it dynamically
        return [
            LlmTaskInput(
                rid=exec_request.orig_instance_id,
                instance_id=exec_request.instance_id,
                block_count=exec_request.block_count,
                seq_len=total_tokens,
                input_tokens=tuple(exec_request.input_token_ids),
                page_ids=tuple(exec_request.page_ids),
                start_position=0 if exec_request.start_position is None else exec_request.start_position,
            )
        ]

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
        assert batch_cfg.token_budget is not None
        token_budget = batch_cfg.token_budget

        prefill_batcher = ExtendAttentionPrefillBatcherProcess(
            fiber=prefill_fiber,
            page_cache=page_cache,
            model_params=batch_cfg.model_params,
            prefill_functions=batch_cfg.prefill_functions,
            program_isolation=batch_cfg.prog_isolation,
            token_budget=token_budget,
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
