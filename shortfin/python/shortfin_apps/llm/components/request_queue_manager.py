# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
import logging
import math
from .config_struct import ModelParams, PagedKVCacheParams
from typing import Optional
from .token_selection_strategy.config import DecodeConfig
from shortfin.interop.fastapi import FastAPIResponder
from shortfin.support.responder import ResponderErrorCodes
from .kvcache.page_pool import PagePool
from .tokenizer import Encoding

logger = logging.getLogger(__name__)


class RequestQueueManager:
    """
    Manages a thread-safe request queue with memory availability checks.
    """

    DEFAULT_MAX_QUEUE_SIZE = 3

    def __init__(
        self,
        *,
        model_params: ModelParams,
        page_pool: PagePool,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ):
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._current_queue_size = 0
        self._current_id = 0
        self._current_tasks = {}

        self.model_params = model_params
        self.page_pool = page_pool
        # Use model_params.decode_batch_sizes to decide actual _max_queue_size
        self._max_queue_size = (
            max(model_params.decode_batch_sizes)
            if model_params.decode_batch_sizes
            else max_queue_size
        )
        logger.debug(f"Initialized with max queue size: {self._max_queue_size}")

    def current_tasks(self) -> list[int]:
        """Returns the IDs of current tasks in the queue."""
        with self._lock:
            return list(self._current_tasks.keys())

    def get_max_queue_size(self) -> int:
        """Returns the maximum queue size."""
        return self._max_queue_size

    def _check_topk_params(self, decode_configs: list[DecodeConfig], responder: FastAPIResponder) -> bool:
        for decode_config in decode_configs:
            exported_topk = self.model_params.top_k
            requested_topk = (
                max(decode_config.num_beams, exported_topk or 1)
                if decode_config.use_beam_search
                else decode_config.top_k
            )

            if not (
                # Argmax
                requested_topk is None
                # CPU-based `beam_search, top_k, and/or top_p`
                or exported_topk is None
                # GPU-based `beam_search, top_k, and/or top_p`
                or exported_topk >= requested_topk
            ):
                logger.error(
                    f"Requested top-k of {requested_topk} larger than exported top-k of {exported_topk}"
                )

                responder.send_error(
                    error_message="Requested top-k larger than exported top-k",
                    code=ResponderErrorCodes.INVALID_REQUEST_ARGS,
                    extra_fields={
                        "exported_topk": exported_topk,
                        "requested_topk": requested_topk,
                    },
                )
                return False

        return True

    def _check_memory_availability(
        self,
        *,
        input_token_ids_len: int,
        available_pages: int,
        decode_config: DecodeConfig,
    ) -> tuple[bool, int]:
        stride = self.model_params.paged_kv_cache.block_seq_stride
        total_beams = decode_config.num_beams
        input_pages = math.ceil(input_token_ids_len / stride)
        copy_pages = total_beams - 1
        output_pages = total_beams * math.ceil(
            decode_config.max_completion_tokens / stride
        )
        needed_pages = input_pages + copy_pages + output_pages

        logger.debug(
            f"Memory check: needed={needed_pages}, available={available_pages}"
        )

        return needed_pages <= available_pages, needed_pages

    def add_to_queue(
        self,
        *,
        decode_configs: list[DecodeConfig],
        input_batch: list[Encoding],
        is_pretokenized: bool,
        responder: FastAPIResponder,
    ) -> Optional[int]:
        """
        Attempts to add a request to the queue.
        Returns: Request ID if successful, None otherwise.
        """

        if not self._check_topk_params(decode_configs, responder):
            return None

        # Step 2: Pre-calculate total needed pages
        stride = self.model_params.paged_kv_cache.block_seq_stride
        available_pages = self.page_pool.get_available_pages_num()
        total_needed_pages = 0

        for index, input_tokens in enumerate(input_batch):
            decode_config = decode_configs[index]
            input_token_ids = input_tokens if is_pretokenized else input_tokens.ids

            input_pages = math.ceil(len(input_token_ids) / stride)
            copy_pages = decode_config.num_beams - 1
            output_pages = decode_config.num_beams * math.ceil(
                decode_config.max_completion_tokens / stride
            )

            total_needed_pages += input_pages + copy_pages + output_pages

        # Step 3: Check if total memory fits
        if total_needed_pages > available_pages:
            responder.send_error(
                error_message="Not enough memory pages available.",
                code=ResponderErrorCodes.KVCACHE_PAGES_FULL,
                extra_fields={
                    "available_page": available_pages,
                    "requested_page": total_needed_pages,
                },
            )
            return None

        request_size = sum(config.num_beams for config in decode_configs)
        with self._lock:
            if self._current_queue_size + request_size > self._max_queue_size:
                responder.send_error(
                    error_message="Server queue is full. Please try again later.",
                    code=ResponderErrorCodes.QUEUE_FULL,
                    extra_fields={
                        "current_size": self._current_queue_size,
                        "max_size": self._max_queue_size,
                    },
                )

                logger.debug(
                    f"Request rejected: {self._current_queue_size} (current) + {request_size} (new) > {self._max_queue_size} (max)."
                )
                return None

            self._current_id += 1
            self._current_queue_size += request_size
            assert self._current_id not in self._current_tasks
            self._current_tasks[self._current_id] = request_size

            logger.debug(
                f"Request added: id={self._current_id}, new queue size={self._current_queue_size}"
            )
            return self._current_id

    def remove_from_queue(self, id: Optional[int]) -> None:
        """
        Remove a request from the queue.

        Args:
            id: The ID of the request to remove
        Raises:
            RuntimeError: If the queue does not have the request ID.
        """
        with self._lock:
            if id not in self._current_tasks:
                error_msg = (
                    f"Remove failed: queue size {self._current_queue_size}, "
                    f"request id {id}"
                )
                logger.debug(error_msg)
                raise RuntimeError(error_msg)

            request_size = self._current_tasks[id]
            del self._current_tasks[id]
            self._current_queue_size -= request_size
            logger.debug(
                f"Request removed: id={id}, new queue size={self._current_queue_size}"
            )
