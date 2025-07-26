# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import shortfin as sf


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

from ...utils import GenerateService
from .request_queue_manager import RequestQueueManager
from .fiber_pool import FiberPool

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
        sysman: LlmSystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: ServerParams,
        program_isolation: str = "per_call",
    ):
        super().__init__(sysman)
        self.name = name
        self.tokenizer = tokenizer

        self.model_params = model_params
        self.server_params = server_params

        self.set_isolation(program_isolation)
        self._initialize_worker_and_fiber()
        self._initialize_page_cache()
        self.queue_manager = RequestQueueManager(model_params=self.model_params)

        self.main_fiber_pool = FiberPool(
            self.sysman, self.queue_manager.get_max_queue_size(), resizable=True
        )

    def _initialize_worker_and_fiber(self):
        self.main_worker = self.sysman.ls.create_worker(f"{self.name}-inference-main-0")
        self.main_fiber = self.sysman.ls.create_fiber(self.main_worker)

        self.prefill_worker = self.sysman.ls.create_worker(
            f"{self.name}-inference-prefill-0"
        )
        self.prefill_fiber = self.sysman.ls.create_fiber(self.prefill_worker)

        self.decode_worker = self.sysman.ls.create_worker(
            f"{self.name}-inference-decode-0"
        )
        self.decode_fiber = self.sysman.ls.create_fiber(self.decode_worker)

        self.devices = self.prefill_fiber.devices_dict.values()

    def _initialize_page_cache(self):
        """Initialize page pool and attention cache."""
        page_pool_config = PagePoolConfig(
            dtype=self.model_params.paged_kv_cache.kv_cache_dtype,
            alloc_page_count=self.model_params.paged_kv_cache.device_block_count,
            paged_kv_block_size_elements=self.model_params.paged_kv_block_size_elements,
            paged_kv_block_size_elements_per_device=self.model_params.paged_kv_cache.paged_kv_block_size_elements_per_device,
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
        component_modules = self.initialize_program_modules("main")
        self.inference_program = self.create_program(
            modules=component_modules, devices=self.sysman.ls.devices
        )
        self.initialize_function_references()

        self.prefill_batcher = PrefillBatcherProcess(
            self.prefill_fiber,
            self.page_cache,
            self.model_params,
            self.prefill_functions,
            self.prog_isolation,
        )

        self.decode_batcher = DecodeBatcherProcess(
            self.decode_fiber,
            self.page_cache,
            self.model_params,
            self.decode_functions,
            self.prog_isolation,
        )

        self.prefill_batcher.launch()
        self.decode_batcher.launch()

    def shutdown(self):
        super().shutdown()
        self.prefill_batcher.shutdown()
        self.decode_batcher.shutdown()
        self.page_cache.shutdown()

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
