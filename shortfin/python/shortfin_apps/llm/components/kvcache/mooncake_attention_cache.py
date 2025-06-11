from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
import time
import math
import heapq
from .page_pool import PagePool, PageInfo
from .base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
    PageAllocation,
)
from .trie_attention_cache import (
    TriePagedAttentionCache,
    TrieNode,
    TriePagedAttentionCacheAllocation,
)
from .mooncake import MooncakeConfig, MooncakeStore


class MooncakePagedAttentionCacheAllocation(TriePagedAttentionCacheAllocation):
    """Allocation for Mooncake paged attention cache.

    Inherits from TriePagedAttentionCacheAllocation to include mooncake store
    specific attributes.
    """

    def __init__(
        self,
        cache: "MooncakePagedAttentionCache",
        tokens: List[int],
        last_cached_node: TrieNode,
        cached_pages: List[PageInfo],
        newly_acquired_pages: List[PageInfo],
    ):
        """Initialize the allocation with a trie node and page info."""
        super().__init__(
            cache, tokens, last_cached_node, cached_pages, newly_acquired_pages
        )

    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        """Publish pages for the given number of tokens, and save the published pages into Mooncake store.
        Args:
            tokens: tokens to publish pages for
            publish_incomplete_page: Whether to publish an incomplete page if it exists
        """
        super().publish_pages_for_tokens(
            tokens, publish_incomplete_page=publish_incomplete_page
        )
        self.send_pages_to_mooncake_store(tokens)

    def send_pages_to_mooncake_store(self, tokens) -> None:
        """Send the pages to Mooncake store for persistent storage.

        Args:
            tokens: list of cached token ids
        """
        # Here we would implement the logic to send the pages to Mooncake store
        # This is a placeholder implementation
        if self.number_of_published_pages == 0:
            print("No pages to publish to Mooncake store.")
            return
        print(
            f"Sending {self.number_of_published_pages} key-value PUTs to Mooncake store."
        )

        for i in range(self.number_of_published_pages):
            page = self._pages[i]
            token_ids = tokens[
                i * self.cache.tokens_per_page : (i + 1) * self.cache.tokens_per_page
            ]
            print(f"Sending page {i} for {len(token_ids)} tokens to Mooncake store.")
            self.cache.send_page_to_mooncake(token_ids, page)


class MooncakePagedAttentionCache(TriePagedAttentionCache):
    """Paged attention cache implementation with Mooncake store support.

    Implements prefix sharing through a trie structure and mooncake store.

    Attributes:
        mooncake_store: mooncake store client for persistent storage
        page_pool: Pool providing page allocations
        tokens_per_page: Number of tokens that fit in each page
    """

    def __init__(
        self, page_pool: PagePool, tokens_per_page: int, mooncake_config_path: str
    ):
        """Initialize the trie cache.

        Args:
            mooncake_config_path: Path to Mooncake configuration file
            page_pool: Pool to allocate pages from
            tokens_per_page: Number of tokens per page

        Raises:
            ValueError: If tokens_per_page <= 0
        """
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")

        super().__init__(page_pool, tokens_per_page)
        self.mooncake_config_path = mooncake_config_path
        mooncake_config = MooncakeConfig.from_json(self.mooncake_config_path)
        self.mooncake_store = MooncakeStore(mooncake_config)
        self.mooncake_keys: Set[str] = set()
        print(f"Mooncake store enabled with config: {self.mooncake_config_path}")

    def send_page_to_mooncake(
        self,
        token_ids: List[int],
        page: PageInfo,
    ) -> None:
        """Send a page to Mooncake store for persistent storage.

        Args:
            token_ids: List of token ids as the key for the page
            page: PageInfo object containing page details
        """
        if not self.mooncake_store:
            raise ValueError("Mooncake store is not initialized")

        key = f"{token_ids[0]}-{token_ids[-1]}"
        print(f"Sending page with key {key} to Mooncake store.")
        value = self.page_pool.get_page_data(page)
        if key in self.mooncake_keys:
            print(f"Page with key {key} already exists in Mooncake store, updating.")
        else:
            print(f"Page with key {key} is new, adding to Mooncake store.")
            self.mooncake_keys.add(key)
            self.mooncake_store.put(key, value)
            print(f"Page with key {key} sent to Mooncake store successfully.")

    def acquire_pages_for_tokens(
        self,
        tokens: List[int],
        extra_token_slots: int = 0,
    ) -> PageAllocation:
        """Acquire pages for a sequence of tokens.

        Attempts to reuse existing cached pages where possible through
        prefix matching, allocating new pages only for the uncached suffix.

        Args:
            tokens: Sequence of tokens needing pages
            extra_token_slots: Additional token slots to allocate beyond tokens

        Returns:
            PageAllocation containing both cached and newly allocated pages

        Raises:
            CacheAllocationFailure: If unable to allocate required pages
        """
        tokens = tuple(tokens)

        cur_node, matched_pages = self._match(tokens)
        cur_node.ref_count.increment()

        n_cached_tokens = len(matched_pages) * self.tokens_per_page
        remaining_length = len(tokens) - n_cached_tokens + extra_token_slots
        n_empty_pages = math.ceil(remaining_length / self.tokens_per_page)

        new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

        if new_pages is not None:
            return MooncakePagedAttentionCacheAllocation(
                cache=self,
                tokens=tokens,
                last_cached_node=cur_node,
                cached_pages=matched_pages,
                newly_acquired_pages=new_pages,
            )

        # Try eviction
        self._evict_pages(n_empty_pages - len(self.page_pool.available_pages))
        new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

        if new_pages is None:
            raise CacheAllocationFailure(
                "Failed to acquire pages even after attempting eviction from LRU leaves"
            )

        return MooncakePagedAttentionCacheAllocation(
            cache=self,
            tokens=tokens,
            last_cached_node=cur_node,
            cached_pages=matched_pages,
            newly_acquired_pages=new_pages,
        )
