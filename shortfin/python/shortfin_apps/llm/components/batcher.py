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

from .host_cache import PrefillHostCacheType, DecodeHostCacheType

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
        functions: dict[int, sf.ProgramFunction],
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
        prefill_functions: dict[int, sf.ProgramFunction],
        program_isolation: str,
        host_cache: PrefillHostCacheType,
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
        self.host_cache = host_cache

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        return PrefillExecutorProcess(
            fiber,
            self.functions,
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
            self.host_cache,
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
        decode_functions: dict[int, sf.ProgramFunction],
        program_isolation: str,
        host_cache: DecodeHostCacheType,
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
        self.host_cache = host_cache

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        return DecodeExecutorProcess(
            fiber,
            self.functions,
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
            self.host_cache,
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
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.seq_stride = seq_stride
        self.exec_requests: list[LlmInferenceExecRequest] = []
        self.page_tables = page_tables
        self.functions = functions
        self.program_isolation = program_isolation
        self.device0 = fiber.device(0)
        self.bs = 0
        self.bsl = 0

    async def get_args(self):
        ...

    async def get_results(
        self,
        logits: sfnp.device_array,
        indices: sfnp.device_array | None,
        req_count: int,
    ):
        ...

    async def _transfer_buffer(self, req_count, buffers):
        ...

    async def run(self):
        try:
            req_bs = len(self.exec_requests)
            seq_stride = self.seq_stride
            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            self.bs = bs
            args, req_count = await self.get_args()

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
            result = await fn(*args, fiber=self.fiber)

            indices = None
            logits = result[0]
            if len(result) > 1:
                indices = result[1]

            # publish cache pages
            for r in self.exec_requests:
                total_tokens = r.start_position + len(r.input_token_ids)
                number_of_complete_pages = total_tokens // seq_stride
                r.publish_allocated_pages(number_of_complete_pages)

            logits, indices = await self._transfer_buffer(
                req_count=req_count, buffers=(logits, indices)
            )

            # Return results.
            await self.get_results(logits, indices, req_count)

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
        host_cache: PrefillHostCacheType,
    ):
        super().__init__(
            name="prefill_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=program_isolation,
        )
        self.host_cache = host_cache

    async def get_args(self):
        seq_stride = self.seq_stride

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        for r in self.exec_requests:
            assert r.start_position == 0

        bsl = max((len(r.input_token_ids)) for r in self.exec_requests)
        self.bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = self.bsl // seq_stride
        req_count = len(self.exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", self.bs, self.bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.
        int_dtype = sfnp.int64
        tokens = sfnp.device_array.for_device(
            self.device0, [self.bs, self.bsl], int_dtype
        )
        seq_lens = sfnp.device_array.for_device(self.device0, [self.bs], int_dtype)
        seq_block_ids = sfnp.device_array.for_device(
            self.device0, [self.bs, block_count], int_dtype
        )

        # Populate tokens
        tokens_key = (self.bs, self.bsl)
        tokens_host_cache = self.host_cache[0]
        if tokens_key not in tokens_host_cache:
            tokens_host_cache[tokens_key] = tokens.for_transfer()
        tokens_host = tokens_host_cache[tokens_key]

        for i in range(self.bs):
            with tokens_host.view(i).map(discard=True) as m:
                m.fill(0)
                if i < req_count:
                    m.items = self.exec_requests[i].input_token_ids
        tokens_host.copy_to(tokens)

        # Populate seq_lens
        seq_lens_host_cache = self.host_cache[1]
        if self.bs not in seq_lens_host_cache:
            seq_lens_host_cache[self.bs] = seq_lens.for_transfer()
        seq_lens_host = seq_lens_host_cache[self.bs]

        with seq_lens_host.map(discard=True) as m:
            m.fill(1)
            m.items = [len(req.input_token_ids) for req in self.exec_requests]
        seq_lens_host.copy_to(seq_lens)

        # Populate cache pages.
        seq_block_ids_key = (self.bs, block_count)
        seq_block_ids_host_cache = self.host_cache[2]
        if seq_block_ids_key not in seq_block_ids_host_cache:
            seq_block_ids_host_cache[seq_block_ids_key] = seq_block_ids.for_transfer()
        seq_block_ids_host = seq_block_ids_host_cache[seq_block_ids_key]
        for i in range(self.bs):
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
        for page_table in self.page_tables:
            args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, indices, req_count):
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = len(req.input_token_ids)
            if req.return_all_logits:
                logits_item = logits.view(i, slice(0, sl))
            else:
                logits_item = logits.view(i, sl - 1)

            index_item = None
            if indices is not None:
                index_item = indices.view(i, sl - 1)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self.exec_requests:
            req.done.set_success()

    async def _transfer_buffer(self, req_count, buffers):
        transfer = any(
            self.exec_requests[i].return_host_array for i in range(req_count)
        )

        if not transfer:
            return buffers

        logits, indices = buffers

        host_logits_key = (self.bs, self.bsl)
        host_logits_cache = self.host_cache[3]
        if host_logits_key not in host_logits_cache:
            logger.debug(
                f"Creating new host logits buffer during prefill. Host key: {host_logits_key}"
            )
            host_logits_cache[host_logits_key] = logits.for_transfer()
        logits_host = host_logits_cache[host_logits_key]

        indices_host = None
        if indices is not None:
            host_indices_cache = self.host_cache[4]
            if host_logits_key not in host_indices_cache:
                logger.debug(
                    f"Creating new host indices buffer during prefill. Host key: {host_logits_key}"
                )
                host_indices_cache[host_logits_key] = indices.for_transfer()
            indices_host = host_indices_cache[host_logits_key]

        # Copy data from device to host
        logits_host.copy_from(logits)
        if indices is not None:
            indices_host.copy_from(indices)

        await self.device0

        return logits_host, (indices_host if indices is not None else None)


class DecodeExecutorProcess(LlmExecutorProcess):
    """Executes a decode batch."""

    def __init__(
        self,
        fiber: Fiber,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        isolation: sf.ProgramIsolation,
        host_cache: DecodeHostCacheType,
    ):
        super().__init__(
            name="decode_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=isolation,
        )
        self.host_cache = host_cache

    async def get_args(self):
        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        seq_stride = self.seq_stride
        bsl = max((1 + len(r.input_token_ids)) for r in self.exec_requests)
        self.bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = self.bsl // seq_stride
        req_count = len(self.exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", self.bs, self.bsl)

        # Prepare inputs.
        # TODO: Better support in shortfin for h2d. The best way to do it is
        # device dependent.
        int_dtype = sfnp.int64
        tokens = sfnp.device_array.for_device(self.device0, [self.bs, 1], int_dtype)
        start_positions = sfnp.device_array.for_device(
            self.device0, [self.bs], int_dtype
        )
        seq_lens = sfnp.device_array.for_device(self.device0, [self.bs], int_dtype)
        seq_block_ids = sfnp.device_array.for_device(
            self.device0, [self.bs, block_count], int_dtype
        )

        # Setup host buffers for transfer:
        tokens_host_cache = self.host_cache[0]
        if self.bs not in tokens_host_cache:
            tokens_host_cache[self.bs] = tokens.for_transfer()
        tokens_host = tokens_host_cache[self.bs]

        seq_lens_host_cache = self.host_cache[1]
        if self.bs not in seq_lens_host_cache:
            seq_lens_host_cache[self.bs] = seq_lens.for_transfer()
        seq_lens_host = seq_lens_host_cache[self.bs]

        start_positions_host_cache = self.host_cache[2]
        if self.bs not in start_positions_host_cache:
            start_positions_host_cache[self.bs] = start_positions.for_transfer()
        start_positions_host = start_positions_host_cache[self.bs]

        seq_block_ids_key = (self.bs, block_count)
        seq_block_ids_host_cache = self.host_cache[3]
        if seq_block_ids_key not in seq_block_ids_host_cache:
            seq_block_ids_host_cache[seq_block_ids_key] = seq_block_ids.for_transfer()
        seq_block_ids_host = seq_block_ids_host_cache[seq_block_ids_key]

        # Populate tokens.
        with tokens_host.map(discard=True) as m:
            m.fill(0)
            vals = []
            for i in range(self.bs):
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
            for i in range(self.bs):
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
        for page_table in self.page_tables:
            args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, indices, req_count):

        # Return results.
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = 1
            if req.return_all_logits:
                logits_item = logits.view(i, slice(0, sl))
            else:
                logits_item = logits.view(i, sl - 1)

            index_item = None
            if indices is not None:
                index_item = indices.view(i, sl - 1)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self.exec_requests:
            req.done.set_success()

    async def _transfer_buffer(self, req_count, buffers):
        transfer = any(
            self.exec_requests[i].return_host_array for i in range(req_count)
        )

        if not transfer:
            return buffers

        logits, indices = buffers

        host_logits_key = (self.bs, self.bsl)
        host_logits_cache = self.host_cache[4]
        if host_logits_key not in host_logits_cache:
            logger.debug(
                f"Creating new host logits buffer. Host key: {host_logits_key}"
            )
            host_logits_cache[host_logits_key] = logits.for_transfer()
        logits_host = host_logits_cache[host_logits_key]

        indices_host = None
        if indices is not None:
            host_indices_cache = self.host_cache[5]
            if host_logits_key not in host_indices_cache:
                logger.debug(
                    f"Creating new host indices buffer during decode. Host key: {host_logits_key}"
                )
                host_indices_cache[host_logits_key] = indices.for_transfer()
            indices_host = host_indices_cache[host_logits_key]

        # Copy data from device to host
        logits_host.copy_from(logits)
        if indices is not None:
            indices_host.copy_from(indices)

        await self.device0

        return logits_host, (indices_host if indices is not None else None)
