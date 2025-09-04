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
from .config import BatchConfig


class BatchingFacade:
    def __init__(self, *, impl: _BatchingEngineImpl):
        self._impl = impl

    def submit(self, exec_request: LlmInferenceExecRequest):
        self._impl.submit(request=exec_request)

    def launch(self):
        self._impl.launch()

    def shutdown(self):
        self._impl.shutdown()

    def reserve_workload(self, *, rid: str, count: int):
        self._impl.reserve_workload(rid=rid, count=count)

    def model_params(self):
        return self._impl.model_params()

    def get_page_cache(self) -> BasePagedAttentionCache:
        return self._impl.get_page_cache()

    @staticmethod
    def build_batcher(
        batch_config: BatchConfig,
        page_cache: BasePagedAttentionCache,
        prefill_fiber: sf.Fiber,  # type: ignore
        decode_fiber: sf.Fiber | None = None,  # type: ignore
    ) -> "BatchingFacade":
        return BatchingFacade(
            impl=_create_impl(
                batch_cfg=batch_config,
                page_cache=page_cache,
                prefill_fiber=prefill_fiber,
                decode_fiber=decode_fiber,
            )
        )
