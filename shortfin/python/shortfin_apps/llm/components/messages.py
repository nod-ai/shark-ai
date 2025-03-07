# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import List

import shortfin as sf
import shortfin.array as sfnp

from .kvcache.base_attention_cache import BasePagedAttentionCache, PageAllocation
from .kvcache.page_pool import PageInfo
from ...utils import InferenceExecRequest

from uuid import uuid4


def listcopy(l):
    return [x for x in l]


class InferencePhase(Enum):
    PREFILL = 1
    DECODE = 2


@dataclass
class LlmInferenceMetrics:
    # NOTE: THIS IS A TEMPORARY CLASS FOR THE DEMO

    prefill_times: List[float]
    decode_times: List[float]
    batcher_pending_times: List[float]
    ttft: float | None = None
    start_time: float | None = None
    end_time: float | None = None

    def add_prefill_time(self, time: float):
        self.prefill_times.append(time)

    def add_decode_time(self, time: float):
        self.decode_times.append(time)

    @property
    def average_prefill_time(self):
        return sum(self.prefill_times) / len(self.prefill_times)

    @property
    def average_decode_time(self):
        return sum(self.decode_times) / len(self.decode_times)

    @property
    def average_batcher_pending_time(self):
        return sum(self.batcher_pending_times) / len(self.batcher_pending_times)

    @property
    def tpot(self):
        total_tokens = len(self.prefill_times) + len(self.decode_times)
        return (self.end_time - self.start_time) / total_tokens

    def set_start_time(self):
        self.start_time = time()

    def set_end_time(self):
        self.end_time = time()

    def set_ttft(self, time: float):
        self.ttft = time - self.start_time

    def replicate_self(self) -> "LlmInferenceMetrics":
        return LlmInferenceMetrics(
            prefill_times=listcopy(self.prefill_times),
            decode_times=listcopy(self.decode_times),
            batcher_pending_times=listcopy(self.batcher_pending_times),
            ttft=self.ttft,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def __str__(self):
        return f"""
            Total Prefills: {len(self.prefill_times)}
            Total Prefill Time: {sum(self.prefill_times)}
            Average Prefill Time: {self.average_prefill_time}
            Total Decodes: {len(self.decode_times)}
            Total Decode Time: {sum(self.decode_times)}
            Average Decode Time: {self.average_decode_time}
            Total Times Batched: {len(self.batcher_pending_times)}
            Total Batcher Pending Times: {sum(self.batcher_pending_times)}
            Average Batcher Pending Time: {self.average_batcher_pending_time}
            TTFT: {self.ttft}
            TPOT: {self.tpot}
            E2E Time: {self.end_time - self.start_time}
        """


class LlmInferenceExecRequest(InferenceExecRequest):
    """Performs a prefill operation."""

    def __init__(
        self,
        phase: InferencePhase,
        input_token_ids: list[int],
        rid: str | None = None,
        llm_inference_metrics: LlmInferenceMetrics | None = None,
    ):
        super().__init__()
        self.phase = phase
        self.start_position: int = 0
        self.input_token_ids = input_token_ids
        self.output_token_ids = []
        self.done = sf.VoidFuture()
        self.rid = rid
        self.instance_id = str(uuid4())
        self.beam_group_id: str | None = None
        self.cumulative_log_prob: float = 0.0
        self.accumulated_normalization: float = 0.0

        # Response control.
        # If True, return all sequence position logits. If False, return only
        # the last.
        self.return_all_logits: bool = False

        # Move the result array to the host and sync to ensure data is
        # available.
        self.return_host_array: bool = True

        # Result logits as [1, sl, d] where 1 is the preserved batch dim,
        # sl is either 1 (not return_all_logits) or >=1 (return_all_logits).
        self.result_logits: sfnp.device_array | None = None

        # Cache pages that have been locked for this request.
        self.cache: BasePagedAttentionCache | None = None
        self.allocation: PageAllocation | None = None

        self.llm_inference_metrics: LlmInferenceMetrics | None = llm_inference_metrics

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = phase
        self.done = sf.VoidFuture()
        self.return_all_logits = False
        self.return_host_array = True
        self.result_logits = None

    def replicate_self(self) -> "LlmInferenceExecRequest":
        new_exec_req = LlmInferenceExecRequest(
            self.phase,
            listcopy(self.input_token_ids),
            self.rid,
        )
        new_exec_req.output_token_ids = listcopy(self.output_token_ids)
        new_exec_req.accumulated_normalization = self.accumulated_normalization
        new_exec_req.start_position = self.start_position
        if self.result_logits is not None:
            result_logits: sfnp.device_array = self.result_logits.for_transfer()
            result_logits.copy_from(self.result_logits)
            new_exec_req.result_logits = result_logits
        new_exec_req.beam_group_id = self.beam_group_id
        new_exec_req.cache = self.cache
        if self.allocation is not None:
            new_exec_req.allocation = self.allocation.replicate_self()
        if self.llm_inference_metrics is not None:
            new_exec_req.llm_inference_metrics = (
                self.llm_inference_metrics.replicate_self()
            )
        return new_exec_req

    def cache_page_indices(self, max_len: int) -> list[int]:
        if not self.allocation:
            return []
        indices = [p.index for p in self.allocation.pages[:max_len]]
        return indices

    def publish_allocated_pages(self, up_to_page_index: int):
        assert self.allocation
        self.allocation.publish_pages_for_tokens(
            self.input_token_ids, publish_incomplete_page=False
        )

    def free_cache_pages(self):
        if self.allocation:
            self.allocation.release_pages()
            self.allocation = None

    def __repr__(self) -> str:
        """
        String representation for logging purposes. It looks like this:

        LlmInferenceExecRequest[phase=P,pos=0,rid=test123,flags=host,input_token_ids=[1, 2, 3, 4]]

        Use
        `logging.debug("Request: %r", request)`
        and not
        `logging.debug(f"Request: {request}")
        to avoid running through this method all the time.
        """
        phase_char = "D" if self.phase == InferencePhase.DECODE else "P"
        flags = []
        if self.return_all_logits:
            flags.append("all")
        if self.return_host_array:
            flags.append("host")
        flags_str = ",".join(flags)
        return f"LlmInferenceExecRequest[phase={phase_char},pos={self.start_position},rid={self.rid},instance_id={self.instance_id},flags={flags_str},input_token_ids={self.input_token_ids}]"
