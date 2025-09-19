# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for kv caches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
import threading
from typing import List, Iterable

from .page_pool import PageInfo, PagePool
from .attention_cache_abstract import CacheInfo


logger = logging.getLogger(__name__)


# exception for when cache allocation failed
class CacheAllocationFailure(Exception):
    pass


class PageAllocation(ABC):
    """Abstract base class for page allocations in the cache."""

    @property
    @abstractmethod
    def pages(self) -> List[PageInfo]:
        """Returns the list of pages that were allocated."""
        pass

    @abstractmethod
    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        """
        Makes pages available to other requests. For details, reference the derived class in trie_attention_cache.py.
        """
        pass

    @abstractmethod
    def release_pages(self) -> None:
        """Releases the allocation's reference to pages."""
        pass

    @abstractmethod
    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        """
        Extends the allocation to include additional tokens. For details, reference the derived class in trie_attention_cache.py.
        """
        pass


class BasePagedAttentionCache:
    """
    Manages lifecycle of pages (using PageInfo as handles).


    Page States:
        Caching - Page can be read by multiple threads
            - Also maintains a reference count
        Writing - Page is being modified by a single owner thread

    Transitions:
        Caching -> Writing: When acquiring an unreferenced LRU leaf page for writing
        Writing -> Caching: When writing is complete and page is released

    Thread Safety:
        - Multiple readers allowed in ReadableCaching state
        - Single writer exclusive access in Writing state
        - Reference counting prevents eviction of in-use pages
    """

    def __init__(
        self,
        page_pool: PagePool,
        tokens_per_page: int,
        use_ref_counts: bool = True,
    ):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page

        # Reference counting
        self.use_ref_counts = use_ref_counts
        self.ref_counts: None | List[int] = (
            None
            if not use_ref_counts
            else [0 for _ in range(len(self.page_pool.attn_page_entries))]
        )
        self._ref_count_lock: None | threading.Lock = (
            None if not use_ref_counts else threading.Lock()
        )

    def shutdown(self):
        available = self.page_pool.available_page_count()
        total = self.page_pool.total_page_count()
        if available != total:
            raise ValueError(f"Pages lost: {total - available} of {total} unfreed")

    def increment_pages(self, pages: List[PageInfo]):
        if not self.use_ref_counts:
            raise RuntimeError(
                "BaseAttentionCache must have use_ref_counts enabled to increment/decrement reference counts."
            )

        with self._ref_count_lock:
            for page in pages:
                self.ref_counts[page.index] += 1

    def decrement_pages(
        self, pages: List[PageInfo], return_empty_pages: bool = False
    ) -> None | List[PageInfo]:
        if not self.use_ref_counts:
            raise RuntimeError(
                "BaseAttentionCache must have use_ref_counts enabled to increment/decrement reference counts."
            )

        with self._ref_count_lock:
            if return_empty_pages:
                empty_pages = []
            for page in pages:
                self.ref_counts[page.index] -= 1
                if return_empty_pages and self.ref_counts[page.index] <= 0:
                    empty_pages.append(page)

        return empty_pages if return_empty_pages else None

    def free_pages(self, pages: List[PageInfo]):
        if not self.use_ref_counts:
            self.page_pool.free_pages(pages)
            return

        pages_to_free = self.decrement_pages(
            pages,
            return_empty_pages=True,
        )

        self.page_pool.free_pages(pages_to_free)

    def fork_pages(self, pages: List[PageInfo]) -> List[PageInfo]:
        new_pages = pages.copy()
        last_page = new_pages.pop(-1)
        new_page = self.page_pool.copy_page(last_page)
        if new_page is None:
            raise CacheAllocationFailure()

        new_pages.append(new_page)
        self.increment_pages(new_pages)
        return BasePagedAttentionCacheAllocation(new_pages, cache=self)

    def allocate(
        self,
        tokens: List[int],
        allocation_block_size: int = 0,
        cache_info: CacheInfo = None,
        lookup: bool = True,
        evict: bool = True,
    ) -> CacheInfo:
        """
        Given a list of tokens, return a CacheInfo object with metadata about the cache allocation.
        """
        token_count = len(tokens)
        pages_needed = allocation_block_size
        if pages_needed == 0:
            pages_needed = math.ceil(token_count / self.tokens_per_page)
        pages = self.page_pool.acquire_free_pages(pages_needed)

        if pages is None:
            msg = (
                f"FATAL CacheAllocationFailure: Failed to allocate {pages_needed} pages from `PagePool`.\n"
                f"Required pages: {pages_needed}, Available pages: {len(self.page_pool.available_pages)}, Total pages: {self.page_pool.config.alloc_page_count}\n"
                f"Consider re-exporting the model with a higher `--device-block-count` value."
            )
            logger.error(msg)
            raise CacheAllocationFailure(msg)

        if self.use_ref_counts:
            self.increment_pages(pages)
        num_tokens = token_count
        if cache_info is not None:
            pages = cache_info.pages + pages
            num_tokens += cache_info.num_tokens
        return CacheInfo(
            num_tokens=num_tokens,
            pages=pages,
            pool=self.page_pool,
            last_cached_node=None,
        )

    def extend_allocation(
        self, tokens, cache_info, *, extra_token_slots=0
    ) -> CacheInfo:
        # assert old tokens are a prefix of incoming tokens
        # if we don't have enough pages to hold the tokens, we need to allocate more pages
        token_count = len(tokens) + extra_token_slots
        pages_needed = math.ceil(token_count / self.tokens_per_page)
        if pages_needed > len(cache_info.pages):
            new_pages = self.page_pool.acquire_free_pages(
                pages_needed - len(cache_info.pages)
            )
            if new_pages is None:
                msg = (
                    f"FATAL CacheAllocationFailure: Failed to allocate {pages_needed - len(self._pages)} pages from `PagePool`.\n"
                    f"Required pages: {pages_needed}, Available pages: {len(self._cache.page_pool.available_pages)}, Total pages: {self._cache.page_pool.config.alloc_page_count}\n"
                    f"Consider re-exporting the model with a higher `--device-block-count` value."
                )
                logger.error(msg)
                raise CacheAllocationFailure(msg)
            if self.use_ref_counts:
                self.increment_pages(new_pages)

            return CacheInfo(
                num_tokens=token_count,
                pages=cache_info.pages + tuple(new_pages),
                pool=self.page_pool,
                last_cached_node=cache_info.last_cached_node,
            )

    def publish_pages_for_tokens(
        self, tokens, cache_info, *, publish_incomplete_page=False
    ) -> CacheInfo:
        return cache_info  # no-op for base class

    def release_pages(self, cache_info: CacheInfo):
        if cache_info is not None:
            self.free_pages(cache_info.pages)
