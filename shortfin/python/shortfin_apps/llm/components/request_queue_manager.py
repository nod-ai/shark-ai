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

logger = logging.getLogger(__name__)


class RequestQueueManager:
    """
    Manages a thread-safe request queue with a maximum size determined by model parameters.
    Also includes memory availability checks
    """

    def __init__(
        self,
        *,
        model_params: ModelParams,
        page_pool: PagePool,
        responder: FastAPIResponder,
        max_queue_size: int = 3,  # Maximum number of requests in queue
    ):
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._current_queue_size = 0
        self._current_id = 0
        self._current_tasks = {}

        self.model_params = model_params
        self.page_pool = page_pool
        self.responder = responder
        # Use model_params.decode_batch_sizes to decide actual _max_queue_size
        if self.model_params.decode_batch_sizes:
            self._max_queue_size = max(self.model_params.decode_batch_sizes)
            logger.debug(f"Max queue size: {self._max_queue_size}")

    def current_tasks(self):
        with self._lock:
            return self._current_tasks.keys()

    def get_current_queue_size(self)
        with self._lock:
            return _current_queue_size

    def add_to_queue(self, *, decode_configs: list[DecodeConfig], input_batch: list[Encoding], is_pretokenized: bool) -> Optional[int]:
        """
        Attempt to add a request to the queue.

        Args:
            decode_configs: The configurations being asked to add to workload

        Returns:
            ID if the request was added successfully, None if the queue is full.
        """
        request_size = sum(config.num_beams for config in decode_configs)

        with self._lock:
            if self._current_queue_size + request_size > self._max_queue_size:
                self.responder.send_error(
                    error_message="Server queue is full. Please try again later.",
                    code=ResponderErrorCodes.QUEUE_FULL,
                    extra_fields={
                        "current_size": self.get_current_queue_size(),
                        "max_size": self.service._max_queue_size,
                    },
                )

                logger.debug(
                    f"Request rejected: {self._current_queue_size} (current) + {request_size} (new) > {self._max_queue_size} (max)."
                )
                return None


        for index, input_tokens in enumerate(input_batch):
            decode_config = decode_configs[index]

            exported_topk = self.model_params.top_k
            requested_topk = (
                        max(decode_config.num_beams, exported_topk or 1)
                    if decode_config.use_beam_search
                    else decode_config.top_k
            )
            
            if not self._check_topk_params(
                exported_topk,
                requested_topk,
            ):
                self.responder.send_error(
                    error_message="Requested top-k larger than exported top-k",
                    code=ResponderErrorCodes.INVALID_REQUEST_ARGS,
                    extra_fields={
                        "exported_topk": exported_topk,
                        "requested_topk": requested_topk,
                    },
                )
                return  None


            input_token_ids = input_tokens if is_pretokenized else input_tokens.ids

            # return error reponse if no pages available
            if not _check_memory_availability(
                input_token_ids_len=len(input_token_ids),
                available_pages=self.page_pool.get_available_pages_num(),
                decode_config=decode_config,
            ):
                self.responder.send_error(
                    error_message="Not enough pages for the new request",
                    code=ResponderErrorCodes.PAGE_FULL,
                    extra_fields={
                        "available_page": available_pages_num,
                        "requested_page": needed_pages_num,
                    },
                )
                return None
                    
            self._current_id += 1
            self._current_queue_size += request_size
            assert self._current_id not in self._current_tasks
            self._current_tasks[self._current_id] = request_size
            logger.debug(f"Added to queue: new queue size {self._current_queue_size}")
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
                f"Removed from queue: new queue size {self._current_queue_size}"
            )

    def _check_memory_availability(
        self,
        *,
        input_token_ids_len: int,
        available_pages: int,
        decode_config: DecodeConfig,
    ) -> bool:
        """
        Check if there is enough memory (paged KV cache) available for the request.

        Args:
            input_token_ids_len: Length of input tokens
            available_pages: Number of available memory pages
            decode_config: Configuration for decoding

        Returns:
            True if enough memory is available, False otherwise.
        """
        stride = self.model_params.paged_kv_cache.block_seq_stride
        total_requested_beams = decode_config.num_beams
        input_pages = math.ceil(input_token_ids_len / stride)
        copy_pages = total_requested_beams - 1

        output_pages_for_one_beam = math.ceil(
            decode_config.max_completion_tokens / stride
        )
        output_pages_total = total_requested_beams * output_pages_for_one_beam
        needed_pages = input_pages + copy_pages + output_pages_total
        logger.debug(
            f"Needed pages for request is {needed_pages}, pages available is {available_pages}"
        )

        return needed_pages <= available_pages
