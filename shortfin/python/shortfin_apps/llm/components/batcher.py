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


class NewWorkItem(sf.Message):
    def __init__(self, count: int = 1):
        super().__init__()
        self.count = count


class DoneWorkItem(sf.Message):
    def __init__(self, count: int = 1):
        super().__init__()
        self.count = count


class LlmBatcherProcess(BatcherProcess):
    """This batcher provides a high-level mechanism for dispatching LLM tasks."""

    STROBE_SHORT_DELAY = 0.010
    STROBE_LONG_DELAY = 0.010

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
        self._current_workitems = 0

        self.program_isolation = program_isolation

    def handle_inference_request(self, request):
        """Handle an inference request."""
        self.pending.add(request)

    async def process_batches(self):
        """Process batches of requests."""
        await self.board_flights()

    def reserve_workitem(self, count):
        self.submit(NewWorkItem(count))

    def complete_workitem(self, count):
        self.submit(DoneWorkItem(count))

    def custom_message(self, msg):
        if isinstance(msg, NewWorkItem):
            self._current_workitems = self._current_workitems + msg.count
            return

        if isinstance(msg, DoneWorkItem):
            self._current_workitems = self._current_workitems - msg.count
            return

        super().custom_message(msg)

    async def board_flights(self):
        await super().board_flights()

    async def board_flights(self):
        waiting_count = len(self.pending)
        if waiting_count == 0:
            self.strobes = 0
            return
        target_size = min(self._current_workitems, self.ideal_batch_size)
        if waiting_count < target_size:
            logger.info("Pending workitems to be enqueued")
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return

        self.strobes = 0
        cache = self.page_cache

        self.board(cache, self.fiber)
        logger.debug("Post boarding cache state: %r", cache)

    def make_process(self, cache: BasePagedAttentionCache, fiber: Fiber):
        ...

    def board_request(self, cache, request: LlmInferenceExecRequest):
        ...

    def board(self, cache: BasePagedAttentionCache, fiber: Fiber):
        # Fill prefill flights.
        pending = self.pending
        if len(pending) == 0:
            return

        exec_process = self.make_process(cache, fiber)

        for request in pending:
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break

            request = self.board_request(cache, request)

            # Can flight this request.
            if request is not None:
                exec_process.exec_requests.append(request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending.remove(flighted_request)
            # And takeoff.
            exec_process.launch()


class PrefillBatcherProcess(LlmBatcherProcess):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.010
    STROBE_LONG_DELAY = 0.010

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        prefill_functions: dict[int, sf.ProgramFunction],
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
            self.functions,
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
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

    STROBE_SHORT_DELAY = 0.0001
    STROBE_LONG_DELAY = 0.0001

    def __init__(
        self,
        fiber: Fiber,
        page_cache: BasePagedAttentionCache,
        model_params: ModelParams,
        decode_functions: dict[int, sf.ProgramFunction],
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
            self.functions,
            self.page_seq_stride,
            cache.page_pool.page_tables,
            self.program_isolation,
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

    async def get_args(self, bs, device0):
        ...

    async def get_results(self, logits, req_count, device0):
        ...

    async def run(self):
        try:
            req_bs = len(self.exec_requests)
            seq_stride = self.seq_stride
            device0 = self.fiber.device(0)
            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            args, req_count = await self.get_args(bs, device0)

            logger.info(
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
            await self.get_results(logits, req_count, device0)

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
    ):
        super().__init__(
            name="prefill_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=program_isolation,
        )

    async def get_args(self, bs, device0):
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
        for page_table in self.page_tables:
            args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, req_count, device0):
        # Return results.
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
                await device0
            else:
                req.result_logits = logits_item
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
    ):
        super().__init__(
            name="decode_process",
            fiber=fiber,
            functions=functions,
            seq_stride=seq_stride,
            page_tables=page_tables,
            program_isolation=isolation,
        )

    async def get_args(self, bs, device0):
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
        for page_table in self.page_tables:
            args.append(sfnp.disable_barrier(page_table))

        return args, req_count

    async def get_results(self, logits, req_count, device0):
        # Return results.
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
                await device0
            else:
                req.result_logits = logits_item
            req.done.set_success()
