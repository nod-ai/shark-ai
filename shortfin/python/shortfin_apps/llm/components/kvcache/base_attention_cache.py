# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base class for kv caches.
"""

from typing import List, Iterable, Protocol
from .page_pool import PageInfo
import math
from abc import ABC, abstractmethod
from .page_pool import PagePool

# logging
import logging

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
    def replicate_self(self) -> "PageAllocation":
        """Replicate the pages of self in a new PageAllocation instance."""
        pass

    @abstractmethod
    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        """
        Extends the allocation to include additional tokens. For details, reference the derived class in trie_attention_cache.py.
        """
        pass


class BasePagedAttentionCacheAllocation(PageAllocation):
    """Represents a page allocation in the cache."""

    def __init__(self, pages: Iterable[PageInfo], cache: "BasePagedAttentionCache"):
        self._pages = tuple(pages)
        self._cache = cache
        self._is_released = False

    @property
    def pages(self) -> List[PageInfo]:
        return list(self._pages)

    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        pass

    def release_pages(self) -> None:
        if self._is_released:
            logger.warning("Releasing already-released allocation")
            return
        self._cache.page_pool.free_pages(self._pages)
        self._is_released = True

    def replicate_self(self) -> "BasePagedAttentionCacheAllocation":
        logger.info(f"Cache State:\n\nPage Pool: {str(self._cache.page_pool)}")
        new_pages = self.pages
        last_page = new_pages.pop(-1)
        for page in new_pages:
            self._cache.page_pool.ref_counts[page.index] += 1
        last_page_copy = self._cache.page_pool.copy_page(last_page)
        if last_page_copy is None:
            raise CacheAllocationFailure()
        new_pages.append(last_page_copy)

        return BasePagedAttentionCacheAllocation(new_pages, self._cache)

    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        # assert old tokens are a prefix of incoming tokens
        # if we don't have enough pages to hold the tokens, we need to allocate more pages
        token_count = len(tokens) + extra_token_slots
        pages_needed = math.ceil(token_count / self._cache.tokens_per_page)
        if pages_needed > len(self._pages):
            new_pages = self._cache.page_pool.acquire_free_pages(
                pages_needed - len(self._pages)
            )
            if new_pages is None:
                raise CacheAllocationFailure()
            self._pages += tuple(new_pages)

    def __rerp__(self) -> str:
        return f"BasePagedAttentionCacheAllocation(pages={self._pages}, cache={self._cache})"


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

    def __init__(self, page_pool: PagePool, tokens_per_page: int):
        self.page_pool = page_pool
        self.tokens_per_page = tokens_per_page

    def acquire_pages_for_tokens(
        self, tokens: List[int], extra_token_slots: int = 1
    ) -> PageAllocation:
        """
        Given a list of tokens, return a list of pages and a start position to continue generation from.

        Parameters:
        - tokens: all the known tokens for this generation request
        - extra_token_slots: number of kvcache slots needed in addition to the ones needed to hold the given tokens.

        In the base implementation, this will just allocate all new pages, but in shared-kv implementations, we will fetch cached pages if applicable.

        The pages are returned in order.

        No token at idx < n_cached_token should be written to. TODO: consider enforcing this.
        """
        token_count = len(tokens)
        pages_needed = math.ceil(token_count / self.tokens_per_page)
        pages = self.page_pool.acquire_free_pages(pages_needed)

        if pages is None:
            raise CacheAllocationFailure()

        return BasePagedAttentionCacheAllocation(pages, cache=self)
