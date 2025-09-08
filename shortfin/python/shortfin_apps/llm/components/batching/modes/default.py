# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
from typing import List, Optional, Tuple, Union


import shortfin as sf
import shortfin.array as sfnp

from shortfin import Fiber

from ...config_struct import ModelParams
from ...device_array_cache import DeviceArrayCache
from ...invocation import (
    LlmTask,
    DecodeTask,
    PrefillTask,
    LlmTaskInput,
    LlmInvocationProcess,
    LlmTaskResponder,
)
from ...kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from ...messages import LlmInferenceExecRequest, InferencePhase
from ...scheduler import Scheduler

from .....utils import BatcherProcess

from ..config import BatchConfig
from ..batching_trait import BatchingTrait

logger = logging.getLogger(__name__)


########################################################################################
# Task Responders
########################################################################################


class PrefillTaskResponder(LlmTaskResponder):
    def __init__(self, exec_requests: List[LlmInferenceExecRequest]) -> None:
        self._exec_requests = exec_requests

    def set_success(
        self, logits: sfnp.device_array, indices: Optional[sfnp.device_array]
    ) -> None:
        """Set the result of the prefill task.

        Args:
            logits (sfnp.device_array): The logits output from the model.
            indices (Optional[sfnp.device_array]): The token indices output from the model.
        """
        exec_requests = self._exec_requests
        for i in range(len(self._exec_requests)):
            req = exec_requests[i]
            sl = len(req.input_token_ids) - 1

            if logits.shape[1] == 1:
                logits_item = logits.view(i)
            else:
                logits_item = logits.view(i, sl)

            index_item = None
            if indices is not None:
                if indices.shape[1] == 1:
                    index_item = indices.view(i)
                else:
                    index_item = indices.view(i, sl)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self._exec_requests:
            req.done.set_success()

    def set_failure(self, exception):
        logger.error(
            f"""Fatal error in Prefill invocation:
            {exception!r}
            """
        )

        for req in self._exec_requests:
            req.result_logits = None
            req.free_cache_pages()
            req.done.set_success()


class DecodeTaskResponder(LlmTaskResponder):
    def __init__(self, exec_requests: List[LlmInferenceExecRequest]) -> None:
        self._exec_requests = exec_requests

    def set_success(
        self, logits: sfnp.device_array, indices: Optional[sfnp.device_array]
    ) -> None:
        exec_requests = self._exec_requests
        for i in range(len(exec_requests)):
            req = exec_requests[i]
            logits_item = logits.view(i, 0)

            index_item = None
            if indices is not None:
                index_item = indices.view(i, 0)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in exec_requests:
            req.done.set_success()

    def set_failure(self, exception):
        logger.error(
            f"""Fatal error in Decode invocation:
            {exception!r}
            """
        )

        for req in self._exec_requests:
            req.result_logits = None
            req.free_cache_pages()
            req.done.set_success()


########################################################################################
# Batcher
########################################################################################


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
        self.page_cache: BasePagedAttentionCache = page_cache
        self.model_params = model_params
        self.functions = functions
        self.pending: set[LlmInferenceExecRequest] = set()
        # TODO: There is no "ideal" batch size. Use prefill/decode dynamic
        # batching in the scheduling algo.
        self.ideal_batch_size: int = ideal_batch_size
        self.page_seq_stride = self.model_params.paged_kv_cache.block_seq_stride
        self.scheduler = Scheduler(ideal_batch_size=self.ideal_batch_size)
        self.array_cache: DeviceArrayCache = DeviceArrayCache(fiber.device(0))

        self.program_isolation = program_isolation

    def handle_inference_request(self, request):
        """Handle an inference request."""
        self.pending.add(request)

    def shutdown(self):
        """Shutdown the batcher process."""
        super().shutdown()
        self.array_cache.free()

    async def process_batches(self):
        """Process batches of requests."""
        await self.board_flights()

    def reserve_workload(self, *, rid, count):
        return self.scheduler.reserve_workload(batcher=self, count=count, rid=rid)

    def custom_message(self, msg):
        if self.scheduler.handle_scheduler(msg):
            return

        super().custom_message(msg)

    async def board_flights(self):
        """Make, schedule, and launch a batch of pending requests."""
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

        page_cache = self.page_cache
        scheduled = []
        for job in to_schedule:
            scheduled = scheduled + job
            self.board(page_cache, self.fiber, job)
            logger.debug("Post boarding cache state: %r", page_cache)

        pending = set(pending) - set(scheduled)
        self.pending = self.pending | pending

    def make_task_inputs(
        self, exec_requests: List[LlmInferenceExecRequest]
    ) -> LlmTaskInput:
        block_count = max(req.block_count for req in exec_requests)
        tokens = [req.input_token_ids for req in exec_requests]
        page_ids = [req.page_ids for req in exec_requests]

        start_positions = None
        if all(req.start_position is not None for req in exec_requests):
            start_positions = [req.start_position for req in exec_requests]

        return LlmTaskInput(
            block_count=block_count,
            seq_stride=self.page_seq_stride,
            input_tokens=tokens,
            page_ids=page_ids,
            start_positions=start_positions,
        )

    def make_task(
        self,
        requests: List[LlmInferenceExecRequest],
        page_cache: BasePagedAttentionCache,
    ) -> LlmTask:
        ...

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "LlmInvocationProcess":
        """Create instance of `LlmInvoker`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB.
        """
        ...

    def board(
        self, page_cache: BasePagedAttentionCache, fiber: Fiber, to_schedule: set
    ):
        """Create and launch an LlmExecutorProcess for the given request batch.

        Args:
            page_cache (BasePagedAttentionCache): KVCache to use for this flight.
            fiber (Fiber): Fiber to use for invocation.
            to_schedule (set): Scheduled requests to be invoked in this flight.
        """
        # Fill prefill flights.
        assert len(to_schedule) > 0
        assert len(to_schedule) <= self.ideal_batch_size

        exec_requests = []
        for request in to_schedule:
            # Can flight this request.
            if request is not None:
                exec_requests.append(request)

        exec_process = self.make_invoker(page_cache, fiber, exec_requests)

        # We've filled our flight. Remove from the boarding area.
        if exec_requests:
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

    def make_task(
        self,
        requests: List[LlmInferenceExecRequest],
        page_cache: BasePagedAttentionCache,
    ) -> LlmTask:
        task_inputs = self.make_task_inputs(requests)
        return PrefillTask(
            task_inputs=task_inputs,
            array_cache=self.array_cache,
            page_tables=page_cache.page_pool.page_tables,
            has_prefill_position=self.model_params.has_prefill_position,
        )

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "LlmInvocationProcess":
        """Create instance of `LlmInvoker`.

        Args:
            page_cache (BasePagedAttentionCache): KVCache instance.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB.
        """
        return LlmInvocationProcess(
            name="prefill_invocation",
            fiber=fiber,
            llm_task=self.make_task(exec_requests, page_cache),
            functions=self.functions,
            program_isolation=self.program_isolation,
            responder=PrefillTaskResponder(exec_requests),
        )


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

    def make_task(
        self,
        requests: List[LlmInferenceExecRequest],
        page_cache: BasePagedAttentionCache,
    ) -> LlmTask:
        task_inputs = self.make_task_inputs(requests)
        return DecodeTask(
            task_inputs=task_inputs,
            array_cache=self.array_cache,
            page_tables=page_cache.page_pool.page_tables,
        )

    def make_invoker(
        self,
        page_cache: BasePagedAttentionCache,
        fiber: Fiber,
        exec_requests: list[LlmInferenceExecRequest],
    ) -> "LlmInvocationProcess":
        """Create instance of `LlmInvoker`.

        This method creates an instance of `LlmInvoker` to handle the
        execution of the decode function for a batch of requests.

        Args:
            page_cache (BasePagedAttentionCache): The KVCache instance to use for this flight.
            fiber (Fiber): Fiber to execute invocation on.
            exec_requests (list[LlmInferenceExecRequest]): Request batch for invocation.

        Returns:
            LlmInvoker: Process to handle execution of VMFB for decode requests.
        """
        return LlmInvocationProcess(
            name="decode_invocation",
            fiber=fiber,
            llm_task=self.make_task(exec_requests, page_cache),
            functions=self.functions,
            program_isolation=self.program_isolation,
            responder=DecodeTaskResponder(exec_requests),
        )


class DefaultBatchingEngine(BatchingTrait):
    def __init__(self, prefill_lane: LlmBatcherProcess, decode_lane: LlmBatcherProcess):
        self.prefill_lane = prefill_lane
        self.decode_lane = decode_lane

    def submit(self, request: LlmInferenceExecRequest):
        if request.phase == InferencePhase.PREFILL:
            self.prefill_lane.submit(request)
        elif request.phase == InferencePhase.DECODE:
            self.decode_lane.submit(request)
        else:
            raise ValueError(
                "Requested unsupported batching lane: Supported only either prefill or decode in default mode."
            )

    def launch(self):
        self.prefill_lane.launch()
        self.decode_lane.launch()

    def shutdown(self):
        self.prefill_lane.shutdown()
        self.decode_lane.shutdown()

    def reserve_workload(self, rid: str, count: int):
        self.decode_lane.reserve_workload(
            rid=rid,
            count=count,
        )

    def get_model_params(self) -> ModelParams:
        return self.prefill_lane.model_params

    @staticmethod
    def create(
        batch_cfg: BatchConfig, page_cache: BasePagedAttentionCache, prefill_fiber: sf.Fiber, decode_fiber: sf.Fiber | None = None  # type: ignore
    ):
        assert (
            decode_fiber is not None
        ), "Request to construct decode batcher, but no fiber supplied"
        prefill_batcher = PrefillBatcherProcess(
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

        return DefaultBatchingEngine(
            prefill_lane=prefill_batcher,
            decode_lane=decode_batcher,
        )
