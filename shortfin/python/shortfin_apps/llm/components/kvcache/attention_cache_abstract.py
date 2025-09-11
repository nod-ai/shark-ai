# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for attention cache components. This module defines the abstract base classes
for attention cache components used in the ShortFin LLM framework. These classes provide a
foundation for implementing various types of attention caches, including those that has distributed storage support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


class CacheStoreAbstract(ABC):
    """
    Abstract base class for attention cache storage.
    This class defines the interface for storing and retrieving attention cache data.
    """

    pass


@dataclass
class CacheInfo:
    """
    Metadata about the allocated cache space.
    - num_tokens: Number of tokens allocated in the cache.
    - pages: The actual pages allocated in the cache.
    - pool: The cache store where this information is stored.
    - last_cached_node: Optional reference to the last cached node, if applicable.
    """

    num_tokens: int
    pages: Any  # This should be a list of PageInfo or similar objects.
    pool: CacheStoreAbstract
    last_cached_node: Any  # Optional reference to the last cached node, if applicable.


@dataclass
class CacheStoreConfig:
    """
    Configuration for the cache store.
    This class holds the hyperparameters and settings for the cache store.
    """

    max_size: int = 1000  # Maximum number of items in the cache
    eviction_policy: str = "LRU"  # Eviction policy (e.g., LRU, FIFO)
    storage_type: str = "in_memory"  # Type of storage (e.g., in_memory, disk)


class AttentionCacheAbstract(ABC):
    """
    Abstract base class for attention cache components.
    This class defines the interface for attention cache components used in the ShortFin LLM framework.
    """

    @abstractmethod
    def allocate(
        self, tokens: List[int], allocation_block_size: int, *args, **kwargs
    ) -> CacheInfo:
        """
        This method should allocate space in the cache for the given tokens and return their indices.
        Parameters:
        - tokens: List of token IDs to allocate space for.
        - allocation_block_size: number of the blocks to allocate for.

        Returns:
        - CacheInfo: An object containing metadata about the allocated cache space.
        """
        pass

    @abstractmethod
    def extend_allocation(
        self, tokens: List[int], cache_info: CacheInfo, extra_token_slots: int
    ) -> CacheInfo:
        """
        This method should extend the allocated cache space for the given tokens by the specified number of extra token slots.
        Parameters:
        - tokens: List of token IDs to extend space for.
        - cache_info: Existing CacheInfo object containing current allocation details.
        - extra_token_slots: Number of additional token slots to allocate.

        Returns:
        - CacheInfo: An updated object containing metadata about the extended cache space.
        """
        pass
