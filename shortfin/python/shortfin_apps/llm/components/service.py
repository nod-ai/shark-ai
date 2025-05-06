# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from dataclasses import dataclass

import shortfin as sf
import shortfin.array as sfnp


from .batcher import (
    PrefillBatcherProcess,
    DecodeBatcherProcess,
    FiberPool,
    MetaFiber,
    initialize_buffer_object,
    LlmBufferObject,
)
from .config_struct import ModelParams, ServerParams
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from .kvcache.page_pool import PagePoolConfig, PagePool
from .manager import LlmSystemManager
from .tokenizer import Tokenizer
from .token_selection_strategy import get_strategy_from_str, is_ref_counted

from ...utils import GenerateService

logger = logging.getLogger(__name__)


class LlmGenerateService(GenerateService):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        instance_num: int,  # Instance number within the service manager
        sysman: LlmSystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_fiber",
        max_queue_size: int = 3,  # Maximum number of requests in queue
    ):
        super().__init__(sysman)
        self.name = name
        self.instance_num = instance_num
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.max_queue_size = max_queue_size
        self.current_queue_size = 0

        self.set_isolation(program_isolation)
        self.initialize_worker_and_fiber()
        self.initialize_queues()
        self.initialize_page_cache()

    def initialize_queues(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes)
            print(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        # print(f"Instance {self.instance_num} adding to queue: {self.current_queue_size} of {self.max_queue_size}")
        if self.current_queue_size >= self.max_queue_size:
            return False
        self.current_queue_size += 1
        return True

    def remove_from_queue(self):
        """Remove a request from the queue."""
        if self.current_queue_size > 0:
            self.current_queue_size -= 1

    def initialize_worker_and_fiber(self):
        num_workers = self.server_params.workers
        fibers_per_worker = self.server_params.fibers_per_worker

        logger.info(
            f"Creating {num_workers} workers, with {fibers_per_worker} fibers per worker..."
        )
        fibers = []
        for i in range(num_workers):
            worker = self.sysman.ls.create_worker(f"{self.name}-inference-{i}")
            for _ in range(fibers_per_worker):
                fiber = self.sysman.ls.create_fiber(worker)
                fibers.append(fiber)

        meta_fibers = []
        for fiber in fibers:
            device_buffer = LlmBufferObject(
                initialize_buffer_object(fibers[0].device(0), self.model_params)
            )
            meta_fiber = MetaFiber(device_buffer, fiber)
            meta_fibers.append(meta_fiber)

        self.fiber_pool = FiberPool(meta_fibers)

        self.devices = fibers[0].devices_dict.values()
        self.main_worker = self.sysman.ls.create_worker(f"{self.name}-inference")
        self.main_fiber = self.sysman.ls.create_fiber(self.main_worker)
        self.prefill_fiber = self.sysman.ls.create_fiber(self.main_worker)
        self.decode_fiber = self.sysman.ls.create_fiber(self.main_worker)

    def initialize_page_cache(self):
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
                use_ref_counts=is_ref_counted(
                    self.server_params.decode_config.token_selection_strategy
                ),
            )
        else:
            raise ValueError(
                f"Unknown prefix_sharing_algorithm {self.server_params.prefix_sharing_algorithm}. Currently only supporting 'trie' and 'none'."
            )

    def start(self):
        component_modules = self.initialize_program_modules("main")
        self.inference_program = self.create_program(
            modules=component_modules, devices=self.sysman.ls.devices
        )
        self.initialize_function_references()

        self.prefill_batcher = PrefillBatcherProcess(
            self.fiber_pool,
            self.page_cache,
            self.model_params,
            self.prefill_functions,
            self.prog_isolation,
        )

        self.decode_batcher = DecodeBatcherProcess(
            self.fiber_pool,
            self.page_cache,
            self.model_params,
            self.decode_functions,
            self.prog_isolation,
        )

        self.prefill_batcher.launch()
        self.decode_batcher.launch()

    def initialize_function_references(self):
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

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  server_params={self.server_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )
