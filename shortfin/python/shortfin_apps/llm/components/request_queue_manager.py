# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
import logging
from .config_struct import ModelParams

logger = logging.getLogger(__name__)

class RequestQueueManager:
    """
    Manages a thread-safe request queue with a maximum size determined by model parameters.
    """

    def __init__(self, model_params: ModelParams):
        """
        Initialize the RequestQueueManager.

        Args:
            model_params: An object with a 'decode_batch_sizes' attribute (list of ints).
        """
        self.model_params = model_params
        self._lock = threading.Lock()
        self.current_queue_size = 0
        self.max_queue_size = 0
        self._initialize_queues()

    def _initialize_queues(self):
        """Initialize the maximum queue size based on model parameters."""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes)
            logger.debug(f"Max queue size set to: {self.max_queue_size}")
        else:
            logger.warning("decode_batch_sizes is empty or None. Max queue size set to 0.")

    def add_to_queue(self, request_size: int) -> bool:
        """
        Attempt to add a request to the queue.

        Args:
            request_size: The size of the request to add.

        Returns:
            True if the request was added successfully, False if the queue is full.
        """
        with self._lock:
            if self.current_queue_size >= self.max_queue_size:
                logger.debug(f"Add failed: queue size {self.current_queue_size}, request size {request_size}")
                return False
            self.current_queue_size += request_size
            logger.debug(f"Added to queue: new queue size {self.current_queue_size}")
            return True

    def remove_from_queue(self, request_size: int) -> bool:
        """
        Remove a request from the queue.

        Args:
            request_size: The size of the request to remove.

        Returns:
            True if the request was removed successfully, False if not enough items in the queue.
        """
        with self._lock:
            if self.current_queue_size >= request_size:
                self.current_queue_size -= request_size
                logger.debug(f"Removed from queue: new queue size {self.current_queue_size}")
                return True
            logger.debug(f"Remove failed: queue size {self.current_queue_size}, request size {request_size}")
            return False
