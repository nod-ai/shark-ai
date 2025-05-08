# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from dataclasses import dataclass
from typing import List

import shortfin as sf
from typing import override
from .batcher import PrefillBatcherProcess, DecodeBatcherProcess, FiberPool
from .config_struct import ModelParams, ServerParams
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from .kvcache.page_pool import PagePoolConfig, PagePool
from .manager import LlmSystemManager
from .service_debug_dumper import SERVICE_DEBUG_DUMPER
from .tokenizer import Tokenizer
from .token_selection_strategy import get_strategy_from_str, is_ref_counted

from ...utils import GenerateService

logger = logging.getLogger(__name__)


class LlmGenerateMultipleStreamService(GenerateService):
    """
    Implementation of a top-level service that parallelizes tasks using more than
    one HIP streams on the GPU, by opening as many IREE HAL devices per physical device.
    Many functions are very similar to an `LlmGenerateService`, but there are some structural
    differences, such as no fiber pool or mulitple worker/fiber support, which make it harder
    to open multiple devices due to duplicate devices in the Scheduler.
    """

    # To invoke VMFB functions on multiple streams, they must be loaded twice with different
    # affinities specified during construction by passing in the device argument.
    prefill_functions: list[dict[int, sf.ProgramFunction]]
    decode_functions: list[dict[int, sf.ProgramFunction]]

    def __init__(
        self,
        *,
        name: str,
        sysman: LlmSystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_call",
        max_queue_size: int | None = None,  # Maximum number of requests in queue
        num_gpu_streams: int = 1,
    ):
        super().__init__(sysman)
        self.name = name
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.max_queue_size = max_queue_size
        self.current_queue_size = 0
        self.num_gpu_streams = num_gpu_streams
        self.prefill_batchers = []
        self.decode_batchers = []
        self.prefill_functions = []
        self.decode_functions = []
        self.fibers = []
        self.inference_program_list: list[sf.Program] = []

        self.main_worker = self.sysman.ls.create_worker("main_worker")
        self.main_fiber = self.sysman.ls.create_fiber(
            self.main_worker, devices=[self.sysman.ls.devices[0]]
        )
        self.set_isolation(program_isolation)
        self.initialize_worker_and_fiber()
        self.initialize_queues()
        self.initialize_page_cache()

    def initialize_queues(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes and not self.max_queue_size:
            # TODO(vinayakdsci): Add smarter handling for cases when queue size
            # is not exactly divisible by number of streams.
            self.max_queue_size = self.num_gpu_streams * max(
                self.model_params.decode_batch_sizes
            )
            logger.info(f"Max queue size: {self.max_queue_size}")
        elif self.max_queue_size:
            logger.info(f"Max queue size: {self.max_queue_size}")

    # TODO(vinayakdsci): This function could probably be moved to the parent class.
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

    def add_to_queue(self) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        if self.current_queue_size >= self.max_queue_size:
            return False
        self.current_queue_size += 1
        return True

    def remove_from_queue(self):
        """Remove a request from the queue."""
        if self.current_queue_size > 0:
            self.current_queue_size -= 1

    def start(self):
        component_modules = self.initialize_program_modules("main")
        for idx in range(self.num_gpu_streams):
            device = self.sysman.ls.devices[idx]
            self.inference_program_list.append(
                self.create_program(component_modules, devices=[device])
            )

        for idx in range(self.num_gpu_streams):
            self.initialize_function_references(idx)
            self.prefill_batchers.append(
                PrefillBatcherProcess(
                    None,
                    self.page_cache,
                    self.model_params,
                    self.prefill_functions[idx],
                    self.prog_isolation,
                    exec_fiber=self.fibers[idx],
                )
            )

            self.decode_batchers.append(
                DecodeBatcherProcess(
                    None,
                    self.page_cache,
                    self.model_params,
                    self.decode_functions[idx],
                    self.prog_isolation,
                    self.fibers[idx],
                )
            )

            self.prefill_batchers[-1].launch()
            self.decode_batchers[-1].launch()

    def initialize_function_references(self, idx: int):
        inference_program = self.inference_program_list[idx]
        prefill_func_map = {}

        for bs in self.model_params.prefill_batch_sizes:
            prefill_func_map[bs] = inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]
        self.prefill_functions.append(prefill_func_map)

        decode_func_map = {}
        for bs in self.model_params.decode_batch_sizes:
            decode_func_map[bs] = inference_program[
                f"{self.model_params.module_name}.decode_bs{bs}"
            ]
        self.decode_functions.append(decode_func_map)

    def initialize_worker_and_fiber(self):
        # While we can have multiple workers/fibers, offloading tasks onto multiple
        # streams through multiple fibers is difficult to reason about/implement when
        # there are more than one logical devices mapped to one physical device.
        # We begin seeing failures like `Duplicate Device in Scheduler` during initialization.
        # TODO: To avoid unnecessary complexity, the code below assigns one fiber and one worker
        # to one HIP stream. This is open to review and should be revisited when it is apparent
        # that a different approach can yield significantly better performance.
        devices = self.sysman.ls.devices
        assert (
            len(devices) == self.num_gpu_streams
        ), f"Expected {self.num_gpu_streams} HAL devices per physical device, found {len(devices)}"

        for device_idx in range(self.num_gpu_streams):
            hal_device = devices[device_idx]
            worker = self.create_worker(hal_device, device_idx)
            self.fibers.append(
                self.sysman.ls.create_fiber(worker, devices=[hal_device])
            )

        self.devices = self.fibers[0].devices_dict.values()


class LlmGenerateService(GenerateService):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
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
        num_gpu_streams: int = 1,
    ):
        super().__init__(sysman)
        del num_gpu_streams
        self.name = name
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.max_queue_size = max_queue_size
        self.current_queue_size = 0

        self.set_isolation(program_isolation)
        self.initialize_worker_and_fiber()
        self.initialize_queues()
        self.initialize_page_cache()

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

        self.fiber_pool = FiberPool(
            fibers,
            fibers,
        )
        self.devices = fibers[0].devices_dict.values()

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

    def initialize_queues(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes) + 2
            print(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        if self.current_queue_size >= self.max_queue_size:
            return False
        self.current_queue_size += 1
        return True

    def remove_from_queue(self):
        """Remove a request from the queue."""
        if self.current_queue_size > 0:
            self.current_queue_size -= 1

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
