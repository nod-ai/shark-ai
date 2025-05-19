# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os

from dataclasses import dataclass
from typing import List


import shortfin as sf
import shortfin.array as sfnp

from shortfin import Fiber

from .scheduler import Scheduler
from ...utils import BatcherProcess

from .config_struct import ModelParams
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
)

from .messages import LlmInferenceExecRequest, InferencePhase
from .service_debug_dumper import SERVICE_DEBUG_DUMPER

logger = logging.getLogger(__name__)


########################################################################################
# Batcher
########################################################################################

import math


class LlmBatcherProcess(BatcherProcess):
    """This batcher provides a high-level mechanism for dispatching LLM tasks."""

    STROBE_SHORT_DELAY = 0.065
    STROBE_LONG_DELAY = 0.065

    def __init__(
        self,
        name: str,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        functions: list[dict[int, sf.ProgramFunction]],
        ideal_batch_size: int,
        program_isolation: str,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.page_cache = page_cache
        self.model_params = model_params
        self.functions = functions
        self.pending: set[LlmInferenceExecRequest] = set()
        # TODO: There is no "ideal" batch size. Use prefill/decode dynamic
        # batching in the scheduling algo.
        self.ideal_batch_size: int = ideal_batch_size
        self.page_seq_stride = self.model_params.paged_kv_cache.block_seq_stride
        self.worker_index = 0
        self.scheduler = Scheduler(ideal_batch_size=self.ideal_batch_size)

        self.program_isolation = program_isolation

    def handle_inference_request(self, request):
        """Handle an inference request."""
        self.pending.add(request)

    async def process_batches(self):
        """Process batches of requests."""
        await self.board_flights()

    def reserve_workitem(self, *, rid, count):
        return self.scheduler.reserve_workitem(batcher=self, count=count, rid=rid)

    def complete_workitem(self, *, rid, count):
        return self.scheduler.release_workitem(batcher=self, count=count, rid=rid)

    def custom_message(self, msg):
        if self.scheduler.handle_scheduler(msg):
            return

        super().custom_message(msg)

    async def board_flights(self):
        await super().board_flights()

    async def board_flights(self):
        # TODO: Add lock on self.pending
        pending = self.pending
        self.pending = set()

        if len(pending) == 0:
            return

        # Determine the requested requests these jobs are for
        rids = set([j.orig_instance_id for j in pending])

        # Group jobs together under their rid
        rid_map = {rid: [] for rid in rids}
        for j in pending:
            rid_map[j.orig_instance_id].append(j)

        to_schedule = self.scheduler.should_execute(rid_map, self.strobes)

        cache = self.page_cache
        scheduled = []
        for job in to_schedule:
            scheduled = scheduled + job
            self.board(cache, self.fiber, job)
            logger.debug("Post boarding cache state: %r", cache)

        pending = set(pending) - set(scheduled)
        self.pending = self.pending | pending

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        ...

    def board_request(self, cache, request: LlmInferenceExecRequest):
        ...

    def board(self, cache: BasePagedAttentionCache, fiber: Fiber, to_schedule: set):
        # Fill prefill flights.
        assert len(to_schedule) > 0
        assert len(to_schedule) <= self.ideal_batch_size

        exec_process = self.make_process(cache, fiber)

        for request in to_schedule:
            request = self.board_request(cache, request)

            # Can flight this request.
            if request is not None:
                exec_process.exec_requests.append(request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            # And takeoff.
            exec_process.launch()


class PrefillBatcherProcess(LlmBatcherProcess):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.065
    STROBE_LONG_DELAY = 0.065

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        prefill_functions: list[dict[int, sf.ProgramFunction]],
        program_isolation: str,
    ):
        super().__init__(
            name="prefill",
            fiber=fiber,
            page_cache=page_cache,
            model_params=model_params,
            functions=prefill_functions,
            ideal_batch_size=max(model_params.prefill_batch_sizes),
            program_isolation=program_isolation,
        )

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        return PrefillExecutorProcess(
            fiber,
            self.functions[self.worker_index],
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
            self.worker_index,
        )

    def board_request(self, cache, request: LlmInferenceExecRequest):
        needed_pages = math.ceil(len(request.input_token_ids) / self.page_seq_stride)
        # allocate kv cache pages
        try:
            allocation = cache.acquire_pages_for_tokens(
                request.input_token_ids,
                extra_token_slots=0,  # prefill needs no extra kvcache slots to write to
            )
        except CacheAllocationFailure:
            logger.debug("Cannot fulfill request for %d pages", needed_pages)
            return None

        logger.debug(f"Successfully acquired allocation: {allocation}")
        request.free_cache_pages()
        request.allocation = allocation

        return request


class DecodeBatcherProcess(LlmBatcherProcess):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.0006
    STROBE_LONG_DELAY = 0.0006

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        decode_functions: list[dict[int, sf.ProgramFunction]],
        program_isolation: str,
    ):
        super().__init__(
            name="decode",
            fiber=fiber,
            page_cache=page_cache,
            model_params=model_params,
            functions=decode_functions,
            ideal_batch_size=max(model_params.decode_batch_sizes),
            program_isolation=program_isolation,
        )

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        return DecodeExecutorProcess(
            fiber,
            self.functions[self.worker_index],
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
            self.worker_index,
        )

    def board_request(self, cache, request: LlmInferenceExecRequest):
        request.allocation.extend_allocation(
            request.input_token_ids, extra_token_slots=1
        )
        return request


########################################################################################
# Inference Executor
########################################################################################


class LlmExecutorProcess(sf.Process):
    """Executes a prefill batch."""

    def __init__(
        self,
        name: str,
        fiber: Fiber,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
        worker_index: int,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.seq_stride = seq_stride
        self.exec_requests: list[LlmInferenceExecRequest] = []
        self.page_tables = page_tables
        self.functions = functions
        self.program_isolation = program_isolation
        self.worker_index = worker_index

    async def get_args(self, bs, device_index):
        ...

    async def get_results(self, logits, req_count, device_index):
        ...

    async def run(self):
        try:
            req_bs = len(self.exec_requests)
            seq_stride = self.seq_stride
            current_worker_index = self.worker_index
            device0 = self.fiber.device(current_worker_index)
            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            args, req_count = await self.get_args(bs, current_worker_index)

            logger.debug(
                "INVOKE %r: %s",
                fn,
                "".join(
                    [
                        (
                            f"\n  {i}: {ary.shape}"
                            if not isinstance(ary, sfnp.disable_barrier)
                            else f"\n  {i}: {ary.delegate().shape}"
                        )
                        for i, ary in enumerate(args)
                    ]
                ),
            )

            # pre-invocation args dump
            if os.getenv("SHORTFIN_DEBUG_LLM_SERVICE", "False").lower() in (
                "true",
                "yes",
                "1",
                "y",
            ):
                await SERVICE_DEBUG_DUMPER.pre_invocation_debug_dump(
                    executor=self, local_vars=locals()
                )

            # Invoke VMFB. Logits are of shape [bs, bsl, d].
            (logits,) = await fn(*args, fiber=self.fiber)

            # publish cache pages
            for r in self.exec_requests:
                total_tokens = r.start_position + len(r.input_token_ids)
                number_of_complete_pages = total_tokens // seq_stride
                r.publish_allocated_pages(number_of_complete_pages)

            # Return results.
            await self.get_results(logits, req_count, current_worker_index)

        except Exception:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()


class PrefillExecutorProcess(LlmExecutorProcess):
    """Executes a prefill batch."""

    def __init__(
        self,
        fiber: Fiber,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
        worker_index: int,
    ):
        super().__init__(
            name="prefill_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=program_isolation,
            worker_index=worker_index,
        )

    async def get_args(self, bs, device_index):
        seq_stride = self.seq_stride

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        for r in self.exec_requests:
            assert r.start_position == 0

        bsl = max((len(r.input_token_ids)) for r in self.exec_requests)
        bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = bsl // seq_stride
        req_count = len(self.exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", bs, bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.
        int_dtype = sfnp.int64
        device0 = self.fiber.device(device_index)
        tokens = sfnp.device_array.for_device(device0, [bs, bsl], int_dtype)
        seq_lens = sfnp.device_array.for_device(device0, [bs], int_dtype)
        seq_block_ids = sfnp.device_array.for_device(
            device0, [bs, block_count], int_dtype
        )

        # Populate tokens.
        tokens_host = tokens.for_transfer()
        for i in range(bs):
            with tokens_host.view(i).map(discard=True) as m:
                m.fill(0)
                if i < req_count:
                    m.items = self.exec_requests[i].input_token_ids
        tokens_host.copy_to(tokens)

        # Populate seq_lens
        seq_lens_host = seq_lens.for_transfer()
        with seq_lens_host.map(discard=True) as m:
            m.fill(1)
            m.items = [len(req.input_token_ids) for req in self.exec_requests]
        seq_lens_host.copy_to(seq_lens)

        # Populate cache pages.
        seq_block_ids_host = seq_block_ids.for_transfer()
        for i in range(bs):
            with seq_block_ids_host.view(i).map(discard=True) as m:
                m.fill(0)
                if i < req_count:
                    m.items = self.exec_requests[i].cache_page_indices(block_count)
        seq_block_ids_host.copy_to(seq_block_ids)

        # V1 args:
        #  prefill:
        #    tokens: [bs, bsl]
        #    seq_lens: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
        args = [tokens, seq_lens, seq_block_ids]
        page_table = self.page_tables[device_index]
        args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, req_count, device_index):
        # Return results
        device0 = self.fiber.device(device_index)
        await_device = False
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = len(req.input_token_ids)
            if req.return_all_logits:
                logits_item = logits.view(i, slice(0, sl))
            else:
                logits_item = logits.view(i, sl - 1)
            if req.return_host_array:
                req.result_logits = logits_item.for_transfer()
                req.result_logits.copy_from(logits_item)
                await_device = True
            else:
                req.result_logits = logits_item

        if await_device:
            await device0

        for req in self.exec_requests:
            req.done.set_success()


class DecodeExecutorProcess(LlmExecutorProcess):
    """Executes a decode batch."""

    def __init__(
        self,
        fiber: Fiber,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        isolation: sf.ProgramIsolation,
        worker_index: int,
    ):
        super().__init__(
            name="decode_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=isolation,
            worker_index=worker_index,
        )

    async def get_args(self, bs, device_index):
        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        seq_stride = self.seq_stride
        bsl = max((1 + len(r.input_token_ids)) for r in self.exec_requests)
        bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = bsl // seq_stride
        req_count = len(self.exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", bs, bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.
        int_dtype = sfnp.int64
        device0 = self.fiber.device(device_index)
        tokens = sfnp.device_array.for_device(device0, [bs, 1], int_dtype)
        start_positions = sfnp.device_array.for_device(device0, [bs], int_dtype)
        seq_lens = sfnp.device_array.for_device(device0, [bs], int_dtype)
        seq_block_ids = sfnp.device_array.for_device(
            device0, [bs, block_count], int_dtype
        )

        # Setup host buffers for transfer:
        tokens_host = tokens.for_transfer()
        seq_lens_host = seq_lens.for_transfer()
        start_positions_host = start_positions.for_transfer()
        seq_block_ids_host = seq_block_ids.for_transfer()

        # Populate tokens.
        with tokens_host.map(discard=True) as m:
            m.fill(0)
            vals = []
            for i in range(bs):
                if i < req_count:
                    vals = vals + self.exec_requests[i].input_token_ids[-1:]
            m.items = vals

        # For decode, populate start_positions and seq_lens.
        with start_positions_host.map(discard=True) as m:
            m.fill(0)
            m.items = [req.start_position for req in self.exec_requests]

        with seq_lens_host.map(discard=True) as m:
            # Pad unused requests.
            m.fill(
                1  # Must pad with a nonzero value because a division by 0 during softmax floods clobber page (page 0) in cache with NaN values.
            )
            m.items = [req.start_position + 1 for req in self.exec_requests]

        # Populate cache pages.
        with seq_block_ids_host.map(discard=True) as m:
            m.fill(0)
            block_ids = []
            for i in range(bs):
                if i < req_count:
                    batch_ids = self.exec_requests[i].cache_page_indices(block_count)
                    block_ids += batch_ids
                    block_ids += [0] * (block_count - len(batch_ids))
            m.items = block_ids

        # Transfer to device memory:
        tokens_host.copy_to(tokens)
        start_positions_host.copy_to(start_positions)
        seq_lens_host.copy_to(seq_lens)
        seq_block_ids_host.copy_to(seq_block_ids)

        # V1 args:
        #  decode:
        #    tokens: [bs, 1]
        #    seq_lens: [bs]
        #    start_positions: [bs]
        #    seq_block_ids: [bs, blocks]
        #    cache_slabs: ...
        args = [tokens, seq_lens, start_positions, seq_block_ids]
        page_table = self.page_tables[device_index]
        args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, req_count, device_index):
        # Return results.
        device0 = self.fiber.device(device_index)
        await_device = False
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = 1
            if req.return_all_logits:
                logits_item = logits.view(i, slice(0, sl))
            else:
                logits_item = logits.view(i, sl - 1)
            if req.return_host_array:
                req.result_logits = logits_item.for_transfer()
                req.result_logits.copy_from(logits_item)
                await_device = True
            else:
                req.result_logits = logits_item

        if await_device:
            await device0

        for req in self.exec_requests:
            req.done.set_success()
