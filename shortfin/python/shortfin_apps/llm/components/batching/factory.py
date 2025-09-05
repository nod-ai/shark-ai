# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Implements a unified batching factory to provide a single
black box batching interface. Internal implementation details
are not leaked.
"""
import logging

import shortfin as sf

from typing import Union

from .batching_trait import BatchingTrait
from .config import BatchConfig, BatchMode
from .modes.default import DefaultBatchingEngine
from ..kvcache.base_attention_cache import BasePagedAttentionCache
from ..messages import LlmInferenceExecRequest
from ..scheduling import UpdateWorkload


logger = logging.getLogger(__name__)


class _BatchingEngineImpl:
    batching_engine: BatchingTrait
    page_cache: BasePagedAttentionCache

    def __init__(
        self,
        batching_engine: BatchingTrait,
        page_cache: BasePagedAttentionCache,
    ):
        self.batching_engine = batching_engine
        self.page_cache = page_cache

    def launch(self):
        self.batching_engine.launch()

    def shutdown(self):
        self.batching_engine.shutdown()

    def get_page_cache(self) -> BasePagedAttentionCache:
        return self.page_cache

    def submit(self, request: Union[LlmInferenceExecRequest, UpdateWorkload]):
        self.batching_engine.submit(request)

    def reserve_workload(self, *, rid: str, count: int):
        self.batching_engine.reserve_workload(rid=rid, count=count)

    def model_params(self):
        return self.batching_engine.get_model_params()


def _create_impl(batch_cfg: BatchConfig, page_cache: BasePagedAttentionCache, prefill_fiber: sf.Fiber, decode_fiber: sf.Fiber | None = None):  # type: ignore
    logger.info(
        f"Initializing Batching Mode: {batch_cfg.mode.name}, "
        f"with Scheduling Mode: {batch_cfg.scheduler_mode.name}"
    )
    if batch_cfg.mode == BatchMode.DEFAULT:
        return _BatchingEngineImpl(
            DefaultBatchingEngine.create(
                batch_cfg=batch_cfg,
                page_cache=page_cache,
                prefill_fiber=prefill_fiber,
                decode_fiber=decode_fiber,
            ),
            page_cache=page_cache,
        )

    raise ValueError(f"Unsupported Batching Mode: {batch_cfg.mode}")
