# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Implements a unified facade to handle batching.
"""

import shortfin as sf

from typing import Callable

from ..kvcache.base_attention_cache import BasePagedAttentionCache
from ..messages import LlmInferenceExecRequest
from .factory import _BatchingEngineImpl, _create_impl
from .config import BatchConfig, Phase


class UnifiedBatcher:
    def __init__(self, *, impl: _BatchingEngineImpl):
        self._impl = impl
        self._impl_prefill_obj = self._impl.prefill_lane.impl_cls
        self._impl_decode_obj = self._impl.decode_lane.impl_cls if self._impl.decode_lane is not None else None

    def submit(self, phase: Phase) -> Callable[[LlmInferenceExecRequest], None] | None:
        fn = None
        if phase == Phase.PREFILL:
            fn = self._impl_prefill_obj.submit # type: ignore
        if phase == Phase.DECODE:
            fn = self._impl_decode_obj.submit # type: ignore
        if fn is None:
            raise ValueError("Unsupported Batching Lane requested")

        return fn

    def launch(self):
        self._impl.launch()

    def shutdown(self):
        self._impl.shutdown()

    def reserve_workload(self, phase: Phase) -> Callable[[int, int], None]:
        fn = None
        if phase == Phase.PREFILL:
            fn = self._impl_prefill_obj.reserve_workload # type: ignore
        if phase == Phase.DECODE:
            fn = self._impl_decode_obj.reserve_workload # type: ignore

        if fn == None:
            raise ValueError(
                "Unknown Batching Lane requested."
            )

        return fn

    def page_cache(self):
        return self._impl.page_cache

    def prefill_engine(self):
        return self._impl_prefill_obj

    def decode_engine(self):
        return self._impl_decode_obj

    @staticmethod
    def build_batcher(
        batch_config: BatchConfig,
        page_cache: BasePagedAttentionCache,
        prefill_fiber: sf.Fiber,  # type: ignore
        decode_fiber: sf.Fiber | None = None,  # type: ignore
    ) -> "UnifiedBatcher":
        return UnifiedBatcher(
            impl=_create_impl(
                batch_cfg=batch_config,
                page_cache=page_cache,
                prefill_fiber=prefill_fiber,
                decode_fiber=decode_fiber,
            )
        )
