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

import shortfin as sf

from .config import BatchConfig, BatchingLane, BatchMode, Phase
from ..batcher import PrefillBatcherProcess, DecodeBatcherProcess
from ..kvcache.base_attention_cache import BasePagedAttentionCache


class _BatchingEngineImpl:
    prefill_lane: BatchingLane
    decode_lane: BatchingLane | None
    page_cache: BasePagedAttentionCache

    def __init__(
        self, prefill_lane: BatchingLane, page_cache: BasePagedAttentionCache, decode_lane: BatchingLane | None = None
    ):
        self.prefill_lane = prefill_lane
        self.decode_lane = decode_lane
        self.page_cache = page_cache

    def launch(self):
        self.prefill_lane.impl_cls.launch()  # type: ignore
        if self.decode_lane is not None:
            self.decode_lane.impl_cls.launch()  # type: ignore

    def shutdown(self):
        self.prefill_lane.impl_cls.shutdown()  # type: ignore
        if self.decode_lane is not None:
            self.decode_lane.impl_cls.shutdown()  # type: ignore


def _create_impl(batch_cfg: BatchConfig, page_cache: BasePagedAttentionCache, prefill_fiber: sf.Fiber, decode_fiber: sf.Fiber | None = None):  # type: ignore
    if batch_cfg.mode == BatchMode.DEFAULT:
        # Construct default prefill batchers and decode batchers and encapsulate as lanes.
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
        prefill_lane = BatchingLane(phase=Phase.PREFILL, impl_cls=prefill_batcher)
        decode_batcher = DecodeBatcherProcess(
            fiber=decode_fiber,
            page_cache=page_cache,
            model_params=batch_cfg.model_params,
            decode_functions=batch_cfg.decode_functions,
            program_isolation=batch_cfg.prog_isolation,
        )
        decode_lane = BatchingLane(phase=Phase.DECODE, impl_cls=decode_batcher)

        return _BatchingEngineImpl(prefill_lane=prefill_lane, decode_lane=decode_lane, page_cache=page_cache)

    raise ValueError(f"Unsupported Batching Mode: {batch_cfg.mode}")
