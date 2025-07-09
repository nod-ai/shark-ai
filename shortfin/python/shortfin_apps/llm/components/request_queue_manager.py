# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from .token_selection_strategy.config import DecodeConfig


class RequestQueueManager:
    """
    Manages a thread-safe request queue with a maximum size determined by model parameters.
    """

    def __init__(self, max_queue_size: int):
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._current_queue_size = 0
        self._current_id = 0
        self._current_tasks = {}

    def current_tasks(self):
        with self._lock:
            return self._current_tasks.keys()

    def add_to_queue(self, decode_configs: list[DecodeConfig]) -> bool:
        """
        Attempt to add a request to the queue.

        Args:
            decode_configs: The configurations being asked to add to workload

        Returns:
            True if the request was added successfully, False if the queue is full.
        """
        request_size = sum(config.num_beams for config in decode_configs)

        with self._lock:
            if self._current_queue_size + request_size > self._max_queue_size:
                logger.debug(
                    f"Add failed: queue size {self._current_queue_size}, request size {request_size}"
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
            request_size: The configurations being removed to workload

        Raises:
            RuntimeError: If the queue does not have enough items to remove.
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
