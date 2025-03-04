# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import os
from pathlib import Path


import shortfin as sf
import shortfin.array as sfnp

from ...utils import ServiceBase, BatcherProcessBase, prog_isolations

from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
    PageAllocation,
)
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from .kvcache.page_pool import PagePoolConfig, PagePool, PageInfo
from .config_struct import ModelParams, ServerParams
from .manager import SystemManager
from .messages import LlmInferenceExecRequest, InferencePhase
from .tokenizer import Tokenizer
from .service_debug_dumper import SERVICE_DEBUG_DUMPER

logger = logging.getLogger(__name__)


class GenerateService(ServiceBase):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_call",
    ):
        super().__init__(sysman)
        self.name = name

        # Application objects.
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.main_worker = sysman.ls.create_worker(f"{name}-inference")
        self.main_fiber = sysman.ls.create_fiber(self.main_worker)

        # Scope dependent objects.
        self.batcher = LlmBatcherProcess(self)
        page_pool_config = PagePoolConfig(
            dtype=model_params.attn_dtype,
            alloc_page_count=model_params.paged_kv_cache.device_block_count,
            paged_kv_block_size_elements=model_params.paged_kv_block_size_elements,
        )
        page_pool = PagePool(
            devices=self.main_fiber.devices_dict.values(), config=page_pool_config
        )
        if server_params.prefix_sharing_algorithm == "trie":
            self.page_cache = TriePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=model_params.paged_kv_cache.block_seq_stride,
            )
        elif server_params.prefix_sharing_algorithm == "none":
            self.page_cache = BasePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=model_params.paged_kv_cache.block_seq_stride,
            )
        else:
            raise ValueError(
                f"Unknown prefix_sharing_algorithm {server_params.prefix_sharing_algorithm}. Currently only supporting 'trie' and 'none'."
            )

        self.program_isolation = prog_isolations[program_isolation]

    def start(self):
        self.inference_program = sf.Program(
            modules=[
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters["main"]
                )
            ]
            + self.inference_modules["main"],
            devices=self.sysman.ls.devices,
            trace_execution=False,
            isolation=self.program_isolation,
        )
        # Resolve prefill entrypoints.
        self.prefill_functions = {}
        for bs in self.model_params.prefill_batch_sizes:
            self.prefill_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]
        # Resolve decode entrypoints.
        self.decode_functions = {}
        for bs in self.model_params.decode_batch_sizes:
            self.decode_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.decode_bs{bs}"
            ]

        # Start persistent processes.
        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )


########################################################################################
# Batcher
########################################################################################

import math


class LlmBatcherProcess(BatcherProcessBase):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.1
    STROBE_LONG_DELAY = 0.25

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.pending_prefills: set[LlmInferenceExecRequest] = set()
        self.pending_decodes: set[LlmInferenceExecRequest] = set()
        # TODO: There is no "ideal" batch size. Use prefill/decode dynamic
        # batching in the scheduling algo.
        self.ideal_batch_size: int = max(service.model_params.prefill_batch_sizes)
        self.page_seq_stride = service.model_params.paged_kv_cache.block_seq_stride

    def handle_inference_request(self, request):
        """Handle an inference request."""
        phase = request.phase
        if phase == InferencePhase.PREFILL:
            self.pending_prefills.add(request)
        elif phase == InferencePhase.DECODE:
            self.pending_decodes.add(request)
        else:
            logger.error("Illegal LlmInferenceExecRequest phase: %r", phase)

    async def process_batches(self):
        """Process batches of requests."""
        await self.board_flights()

    async def board_flights(self):
        waiting_count = len(self.pending_prefills) + len(self.pending_decodes)
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0
        cache = self.service.page_cache

        # TODO: This is a very naive cache management algorithm. Burn with fire
        # and implement a real one.
        self.board_prefills(cache)
        self.board_decodes(cache)

        # For now, kill anything that is left.
        for prefill_request in self.pending_prefills:
            prefill_request.done.set_success()
        self.pending_prefills.clear()
        logger.debug("Post boarding cache state: %r", cache)

    def board_prefills(self, cache: BasePagedAttentionCache):
        # Fill prefill flights.
        pending_prefills = self.pending_prefills
        if len(pending_prefills) == 0:
            return
        exec_process = InferenceExecutorProcess(
            self.service,
            InferencePhase.PREFILL,
            self.page_seq_stride,
            cache.page_pool.page_tables,
        )
        for prefill_request in pending_prefills:
            assert prefill_request.phase == InferencePhase.PREFILL
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            needed_pages = math.ceil(
                len(prefill_request.input_token_ids) / self.page_seq_stride
            )
            # allocate kv cache pages
            try:
                allocation = cache.acquire_pages_for_tokens(
                    prefill_request.input_token_ids,
                    extra_token_slots=0,  # prefill needs no extra kvcache slots to write to
                )
            except CacheAllocationFailure:
                logger.debug("Cannot fulfill request for %d pages", needed_pages)
                continue
            logger.debug(f"Successfully acquired allocation: {allocation}")
            prefill_request.free_cache_pages()
            prefill_request.allocation = allocation

            # Can flight this request.
            exec_process.exec_requests.append(prefill_request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_prefills.remove(flighted_request)
            # And takeoff.
            exec_process.launch()

    def board_decodes(self, cache: BasePagedAttentionCache):
        # Fill decode flights.
        pending_decodes = self.pending_decodes
        if len(pending_decodes) == 0:
            return
        exec_process = InferenceExecutorProcess(
            self.service,
            InferencePhase.DECODE,
            self.page_seq_stride,
            cache.page_pool.page_tables,
        )
        for decode_request in pending_decodes:
            assert decode_request.phase == InferencePhase.DECODE
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            decode_request.allocation.extend_allocation(
                decode_request.input_token_ids, extra_token_slots=1
            )

            # Can flight this request.
            exec_process.exec_requests.append(decode_request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_decodes.remove(flighted_request)
            # And takeoff.
            exec_process.launch()


########################################################################################
# Inference Executor
########################################################################################


class InferenceExecutorProcess(sf.Process):
    """Executes a prefill or decode batch."""

    def __init__(
        self,
        service: GenerateService,
        phase: InferencePhase,
        seq_stride: int,
        page_tables,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.phase = phase
        self.seq_stride = seq_stride
        self.exec_requests: list[LlmInferenceExecRequest] = []
        self.page_tables = page_tables

    async def run(self):
        try:
            is_decode = self.phase == InferencePhase.DECODE
            req_bs = len(self.exec_requests)
            seq_stride = self.seq_stride
            # Select an entrypoint for the batch.
            if is_decode:
                entrypoints = self.service.decode_functions
            else:
                entrypoints = self.service.prefill_functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            # Compute block sequence length as maximum sequence length, rounded
            # up to the seq_stride.
            if self.phase == InferencePhase.PREFILL:
                for r in self.exec_requests:
                    assert r.start_position == 0

            extra_token_slots = 1 if is_decode else 0
            bsl = max(
                (extra_token_slots + len(r.input_token_ids)) for r in self.exec_requests
            )
            bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
            block_count = bsl // seq_stride
            req_count = len(self.exec_requests)
            logger.debug("Prefill bs=%d, bsl=%d", bs, bsl)

            # Prepare inputs.
            # TODO: Better support in shortfin for h2d. The best way to do it is
            # device dependent.
            device0 = self.fiber.device(0)
            int_dtype = sfnp.int64
            if is_decode:
                tokens = sfnp.device_array.for_device(device0, [bs, 1], int_dtype)
                start_positions = sfnp.device_array.for_device(device0, [bs], int_dtype)
            else:
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
                        if self.phase == InferencePhase.PREFILL:
                            m.items = self.exec_requests[i].input_token_ids
                        elif self.phase == InferencePhase.DECODE:
                            m.items = self.exec_requests[i].input_token_ids[-1:]
            tokens_host.copy_to(tokens)

            # For prefill, populate seq_lens
            if self.phase == InferencePhase.PREFILL:
                seq_lens_host = seq_lens.for_transfer()
                with seq_lens_host.map(discard=True) as m:
                    m.fill(1)
                    m.items = [len(req.input_token_ids) for req in self.exec_requests]
                seq_lens_host.copy_to(seq_lens)

            # For decode, populate start_positions and seq_lens.
            if self.phase == InferencePhase.DECODE:
                start_positions_host = start_positions.for_transfer()
                with start_positions_host.map(discard=True) as m:
                    m.fill(0)
                    m.items = [req.start_position for req in self.exec_requests]
                start_positions_host.copy_to(start_positions)

                seq_lens_host = seq_lens.for_transfer()
                with seq_lens_host.map(discard=True) as m:
                    # Pad unused requests.
                    m.fill(
                        1  # Must pad with a nonzero value because a division by 0 during softmax floods clobber page (page 0) in cache with NaN values.
                    )
                    m.items = [req.start_position + 1 for req in self.exec_requests]
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
            #  decode:
            #    tokens: [bs, 1]
            #    seq_lens: [bs]
            #    start_positions: [bs]
            #    seq_block_ids: [bs, blocks]
            #    cache_slabs: ...
            if is_decode:
                args = [tokens, seq_lens, start_positions, seq_block_ids]
            else:
                args = [tokens, seq_lens, seq_block_ids]
            for page_table in self.page_tables:
                args.append(sfnp.disable_barrier(page_table))
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
            for i in range(req_count):
                req = self.exec_requests[i]
                sl = 1 if is_decode else len(req.input_token_ids)
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

        except Exception:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()
