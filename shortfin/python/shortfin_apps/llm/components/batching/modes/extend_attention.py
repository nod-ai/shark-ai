# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import List

import shortfin as sf

from shortfin import Fiber

from ..batching_trait import BatchingTrait
from ..config import BatchConfig
from ...config_struct import ModelParams
from ...invocation import (
    ExtendAttentionPrefillTask,
    LlmInvocationProcess,
    LlmTask,
    LlmTaskInput,
)
from ...kvcache.base_attention_cache import BasePagedAttentionCache
from ...messages import InferencePhase, LlmInferenceExecRequest
from ...scheduler import ExtendAttentionScheduler

from .default import (
    LlmBatcherProcess,
    PrefillTaskResponder,
    DecodeBatcherProcess,
)

logger = logging.getLogger(__name__)


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

        logger.info(
            f"ExtendAttention make_task_inputs: input_token_ids={exec_request.input_token_ids}"
        )

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
                start_position=0
                if exec_request.start_position is None
                else exec_request.start_position,
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
        decode_fiber: sf.Fiber,
    ):
        """Create an extend-attention batching engine."""

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
