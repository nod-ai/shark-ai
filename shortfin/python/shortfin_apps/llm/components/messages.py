# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import shortfin as sf
import shortfin.array as sfnp

from .kvcache.base_attention_cache import BasePagedAttentionCache, PageAllocation
from .kvcache.page_pool import PageInfo
from ...utils import InferenceExecRequest


class InferencePhase(Enum):
    PREFILL = 1
    DECODE = 2


class LlmInferenceExecRequest(InferenceExecRequest):
    """Performs a prefill operation."""

    def __init__(self, phase: InferencePhase, input_token_ids: list[int], rid=None):
        super().__init__()
        self.phase = phase
        self.start_position: int = 0
        self.input_token_ids = input_token_ids
        self.done = sf.VoidFuture()
        self.rid = rid

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
        self._cache: BasePagedAttentionCache | None = None
        self.allocation: PageAllocation | None = None

    def reset(self, phase: InferencePhase):
        """Resets all per request state in preparation for an subsequent execution."""
        self.phase = phase
        self.done = sf.VoidFuture()
        self.return_all_logits = False
        self.return_host_array = True
        self.result_logits = None

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
        return f"LlmInferenceExecRequest[phase={phase_char},pos={self.start_position},rid={self.rid},flags={flags_str},input_token_ids={self.input_token_ids}]"
