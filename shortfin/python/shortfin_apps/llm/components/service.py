# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from dataclasses import dataclass
from typing import List
from threading import Lock
import shortfin as sf

from .stream_manager import StreamManager

from .batcher import PrefillBatcherProcess, DecodeBatcherProcess
from .config_struct import ModelParams, ServerParams
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from .kvcache.page_pool import PagePoolConfig, PagePool
from .manager import LlmSystemManager
from .service_debug_dumper import SERVICE_DEBUG_DUMPER
from .tokenizer import Tokenizer
from .token_selection_strategy import is_multi_response
from .request_queue_manager import RequestQueueManager

from ...utils import (
    GenerateService,
    LLM_DISAGGREGATED_DECODE_DEVICE_IDX,
    LLM_DISAGGREGATED_PREFILL_DEVICE_IDX,
)
from .fiber_pool import FiberPool

logger = logging.getLogger(__name__)


class LlmGenerateService(GenerateService):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program | list[sf.Program]
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        sysman: LlmSystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_call",
        max_queue_size: int = 3,  # Maximum number of requests in queue
    ):
        super().__init__(sysman)
        self.name = name
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.disaggregate = server_params.disaggregate
        self.max_queue_size = max_queue_size
        # Use model_params.decode_batch_sizes to decide actual max_queue_size
        self._initialize_max_queue_size()
        self.main_fiber_pool = FiberPool(
            self.sysman, self.max_queue_size, resizable=True
        )

        self.set_isolation(program_isolation)
        self.queue_manager = RequestQueueManager(self.max_queue_size)
        self.current_queue_size = 0
        self.set_isolation(program_isolation)
        self._stream_manager = StreamManager(
            self.sysman,
            self.max_queue_size,
        )
        self.main_fiber_pool = self._stream_manager.fiber_pool()
        (
            self.prefill_fiber,
            self.decode_fiber,
            self.prefill_exec_fiber,
            self.decode_exec_fiber,
            self.main_fiber,
        ) = self._stream_manager.construct_main_fibers()

        self.devices = self.prefill_fiber.devices_dict.values()

        self._initialize_page_cache()

    def _initialize_max_queue_size(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes)
            logger.debug(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self, num_beams: int) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        with self._lock:
            if self.current_queue_size >= self.max_queue_size:
                return False
            self.current_queue_size += num_beams
            logger.debug(f"Adding to queue, queue size: {self.current_queue_size}")
            return True

    def remove_from_queue(self, num_beams: int):
        """Remove a request from the queue."""
        with self._lock:
            if self.current_queue_size >= num_beams:
                self.current_queue_size -= num_beams
                logger.debug(
                    f"Removing from queue, queue size: {self.current_queue_size}"
                )

    def _initialize_page_cache(self):
        """Initialize page pool and attention cache."""
        page_pool_config = PagePoolConfig(
            dtype=self.model_params.paged_kv_cache.kv_cache_dtype,
            alloc_page_count=self.model_params.paged_kv_cache.device_block_count,
            paged_kv_block_size_elements=self.model_params.paged_kv_block_size_elements,
        )
        page_pool = PagePool(devices=self.devices, config=page_pool_config)

        if self.server_params.prefix_sharing_algorithm == "trie":
            self.page_cache = TriePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=self.model_params.paged_kv_cache.block_seq_stride,
            )
        elif self.server_params.prefix_sharing_algorithm == "none":
            self.page_cache = BasePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=self.model_params.paged_kv_cache.block_seq_stride,
            )
        else:
            raise ValueError(
                f"Unknown prefix_sharing_algorithm {self.server_params.prefix_sharing_algorithm}. Currently only supporting 'trie' and 'none'."
            )

    def start(self):
        self._stream_manager.load_program_modules(self)

        self.prefill_batcher = PrefillBatcherProcess(
            self.prefill_fiber,
            self.page_cache,
            self.model_params,
            self.prefill_functions,
            self.prog_isolation,
            exec_fiber=self.prefill_exec_fiber,
        )

        self.decode_batcher = DecodeBatcherProcess(
            self.decode_fiber,
            self.page_cache,
            self.model_params,
            self.decode_functions,
            self.prog_isolation,
            exec_fiber=self.decode_exec_fiber,
        )

        self.prefill_batcher.launch()
        self.decode_batcher.launch()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  server_params={self.server_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n",
            f"  disaggregated={self.disaggregate}",
            f")",
        )
