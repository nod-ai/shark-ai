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
    ):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page
        self._allocated_pages: List[
            PageInfo
        ] = []  # global allocated page pool that contains all un-tracked pages

    def shutdown(self):
        self.page_pool.free_pages(self._allocated_pages)
        available = self.page_pool.available_page_count()
        total = self.page_pool.total_page_count()
        if available != total:
            raise ValueError(f"Pages lost: {total - available} of {total} unfreed")

    def free_pages(self, pages: List[PageInfo]):
        self.page_pool.free_pages(pages)

    def fork_pages(self, tokens: list[int], cache_info: CacheInfo) -> CacheInfo:
        new_pages = cache_info.pages.copy()
        last_page = new_pages.pop(-1)
        new_page = self.page_pool.copy_page(last_page)
        if new_page is None:
            raise CacheAllocationFailure()

        new_pages.append(new_page)
        cache_info.pages = new_pages
        cache_info.tokens.extend(tokens)
        cache_info.num_tokens += len(tokens)
        return cache_info

    def lookup(self, tokens: List[int]) -> CacheInfo:
        return CacheInfo(
            num_tokens=0,
            tokens=[],
            pages=[],
            pool=self.page_pool,
            last_cached_node=None,
        )

    def get_allocated_pages(self, page_ids: List[int]) -> List[PageInfo]:
        pages = []
        for page in self._allocated_pages:
            if page.index in page_ids:
                pages.append(page)
        return pages

    def allocate(
        self,
        tokens: List[int],
        cache_info: CacheInfo = None,
        allocation_block_size: int = 0,
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

        num_tokens = token_count
        if cache_info is not None:
            pages = cache_info.pages + pages
            num_tokens += cache_info.num_tokens
        self._allocated_pages.extend(pages)
        return CacheInfo(
            num_tokens=num_tokens,
            tokens=tokens,
            pages=pages,
            pool=self.page_pool,
            last_cached_node=None,
        )

    def publish_pages_for_tokens(
        self, cache_info, *, publish_incomplete_page=False
    ) -> CacheInfo:
        return cache_info  # no-op for base class

    def release_pages(self, cache_info: CacheInfo):
        if cache_info is not None:
            self.free_pages(cache_info.pages)

    def free_allocated_pages(self, page_ids: List[int]):
        pages = []
        for page in self._allocated_pages:
            if page.index in page_ids:
                pages.append(page)
        self.free_pages(pages)
