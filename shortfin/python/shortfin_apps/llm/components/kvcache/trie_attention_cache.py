from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from threading import Lock
import time
import math
import heapq
from copy import deepcopy
from .page_pool import PagePool, PageInfo
from .base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
    PageAllocation,
)
from .kvcache_utils import RefCount
from .attention_cache_abstract import CacheInfo

import logging

logger = logging.getLogger(__name__)


@dataclass
class TrieNode:
    """Node of the block trie for paged attention cache.

    Each node represents a page of tokens in the cache, with edges representing
    token sequences that can follow. This allows prefix sharing between sequences
    that have common prefixes.

    Attributes:
        tokens: Tuple of tokens stored in this node's page
        page: PageInfo object containing the actual cache page
        children: Dict mapping token sequences to child nodes
        parent: Parent node in the trie (None for root)
        ref_count: Number of active references to this node
        access_time: Last access timestamp for LRU eviction
    """

    tokens: Tuple[int, ...]
    page: PageInfo
    children: Optional[Dict[Tuple[int, ...], "TrieNode"]] = None
    parent: Optional["TrieNode"] = None
    ref_count: RefCount = None
    access_time: float = 0.0

    def __post_init__(self) -> None:
        """Initialize children dict and access time if not provided."""
        if self.children is None:
            self.children = {}
        self.access_time = time.monotonic()
        self.ref_count = RefCount()

    def create_child(self, tokens: Tuple[int, ...], page: PageInfo) -> "TrieNode":
        """Create a new child node with the given tokens and page.

        Args:
            tokens: Sequence of tokens for the new node
            page: PageInfo for the new node's cache page

        Returns:
            The newly created child node
        """
        new_node = TrieNode(tokens=tokens, page=page, parent=self)
        self.children[tokens] = new_node
        return new_node

    def unlink(self) -> None:
        """Remove this node from its parent's children."""
        if self.parent is not None:
            del self.parent.children[self.tokens]
            self.parent = None

    def __hash__(self) -> int:
        """Nodes are uniquely identified by their memory address."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Nodes are equal only if they are the same object."""
        return self is other

    def __lt__(self, other):
        """Sort nodes by their memory address."""
        return id(self) < id(other)


@dataclass
class TrieCacheInfo(CacheInfo):
    """Metadata about the trie-based cache allocation.

    Contains information about the tokens, pages, and last cached node.

    Attributes:
        tokens: List of tokens in the allocation
        last_cached_node: Last node in the trie that was cached
        cached_pages: List of pages that were already cached
        newly_acquired_pages: List of pages that were newly acquired for this allocation
        number_of_published_pages: Number of pages that have been published to the cache
    """

    tokens: List[int]
    number_of_published_pages: int
    last_cached_node: TrieNode


class TriePagedAttentionCache(BasePagedAttentionCache):
    """Trie-based paged attention cache implementation.

    Implements prefix sharing through a trie structure where each node
    represents a page of tokens. Common prefixes between sequences share
    the same nodes/pages, reducing memory usage.

    Attributes:
        root: Root node of the trie
        leaves: Set of leaf nodes for efficient eviction
        page_pool: Pool providing page allocations
        tokens_per_page: Number of tokens that fit in each page
    """

    def __init__(self, page_pool: PagePool, tokens_per_page: int):
        """Initialize the trie cache.

        Args:
            page_pool: Pool to allocate pages from
            tokens_per_page: Number of tokens per page

        Raises:
            ValueError: If tokens_per_page <= 0
        """
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")

        super().__init__(page_pool, tokens_per_page, use_ref_counts=False)

        # Create root node with dummy page
        dummy_page = PageInfo(
            index=0,  # Root uses reserved index 0
            pool=self.page_pool,
        )
        self.root = TrieNode(tokens=tuple(), page=dummy_page)
        self.leaves: Set[TrieNode] = set()
        self._lock: Lock = Lock()
        self._allocated_pages: List[PageInfo] = []

    def fork_pages(self, pages: List[PageInfo], tokens: list[int]) -> TrieCacheInfo:
        """Fork a sequence of pages into the trie.

        Share prefixes with existing nodes till N-1 tokens, then create a new node
        for the last token block. This allows sharing of common prefixes


        Args:
            pages: List of PageInfo objects to fork into the trie

        Returns:
            TrieCacheInfo containing both cached and newly allocated pages
        """
        with self._lock:
            curr, matched_pages = self.match(tokens)
            curr.ref_count.increment()

            n_cached_tokens = len(matched_pages) * self.tokens_per_page
            if n_cached_tokens >= len(tokens):
                # If all tokens are already cached, no need to fork
                return TrieCacheInfo(
                    tokens=list(tokens),
                    num_tokens=len(tokens),
                    last_cached_node=curr,
                    pages=matched_pages + [],
                    number_of_published_pages=len(matched_pages),
                    pool=self.page_pool,
                )

            new_page = self.page_pool.copy_page(pages[-1])
            if new_page is None:
                self._evict_pages(1)
                new_page = self.page_pool.copy_page(pages[-1])

            return TrieCacheInfo(
                tokens=list(tokens),
                num_tokens=len(tokens),
                last_cached_node=curr,
                pages=matched_pages + [new_page],
                number_of_published_pages=len(matched_pages),
                pool=self.page_pool,
            )

    def match(self, tokens: List[int]) -> Tuple[TrieNode, List[PageInfo]]:
        """
        Find the longest prefix match in the trie.

        Walks the trie following the token sequence as far as possible,
        collecting matched pages along the way.

        Args:
            tokens: Sequence of tokens to match

        Returns:
            Tuple of (last matched node, list of matched pages, length of last matched token block)
        """
        tokens = tuple(tokens)
        matched_pages = []
        cur = self.root

        for i in range(0, len(tokens), self.tokens_per_page):
            token_block = tokens[i : i + self.tokens_per_page]

            if token_block not in cur.children:
                break
            cur = cur.children[token_block]
            cur.access_time = time.monotonic()
            matched_pages.append(cur.page)

        return cur, matched_pages

    def evict_pages(self, max_pages: int) -> int:
        """Evict up to max_pages pages using LRU strategy.

        Evicts from unreferenced leaf nodes first, working up the trie
        as nodes become childless.

        Args:
            max_pages: Maximum number of pages to evict

        Returns:
            Number of pages actually evicted
        """
        pages_to_evict = []
        # Initialize heap with unreferenced leaves
        unused_leaf_heap = [
            (leaf.access_time, leaf)
            for leaf in self.leaves
            if leaf.ref_count.is_empty()
        ]
        heapq.heapify(unused_leaf_heap)

        # Evict least recently used nodes
        while unused_leaf_heap and len(pages_to_evict) < max_pages:
            _, leaf = heapq.heappop(unused_leaf_heap)
            pages_to_evict.append(leaf.page)
            parent = leaf.parent

            leaf.unlink()
            self.leaves.remove(leaf)

            # If parent becomes childless, it becomes a leaf
            if (
                parent is not self.root
                and not parent.children
                and parent not in self.leaves
            ):
                self.leaves.add(parent)
                if parent.ref_count.is_empty():
                    heapq.heappush(unused_leaf_heap, (parent.access_time, parent))

        if pages_to_evict:
            logger.debug(
                f"TriePagedAttentionCache: Released allocated pages in evict_pages {[p.index for p in pages_to_evict]}"
            )
            self.page_pool.free_pages(pages_to_evict)

        return len(pages_to_evict)

    def allocate(
        self,
        tokens: List[int],
        allocation_block_size: int = 0,
        cache_info: TrieCacheInfo = None,
        lookup: bool = True,
        evict: bool = True,
    ) -> TrieCacheInfo:
        """Acquire pages for a sequence of tokens.

        Attempts to reuse existing cached pages where possible through
        prefix matching, allocating new pages only for the uncached suffix.

        Args:
            tokens: Sequence of tokens needing pages
            allocation_block_size: number of pages to allocate at once, not used if it is 0
            lookup: Whether to look up existing tokens in the cache.
            evict: Whether to evict old tokens if the cache is full.

        Returns:
            PageAllocation containing both cached and newly allocated pages

        Raises:
            CacheAllocationFailure: If unable to allocate required pages
        """
        with self._lock:
            tokens = tuple(tokens)
            n_empty_pages = 0
            cached_pages = []
            pages = []
            cur_node = self.root
            if lookup:
                cur_node, matched_pages = self.match(tokens)
                logger.debug(
                    f"TriePagedAttentionCache: Lookup found {len(matched_pages)} cached pages for token length {len(tokens)}"
                )

                cached_pages = matched_pages
                n_cached_tokens = 0
                if matched_pages:
                    n_cached_tokens = len(matched_pages) * self.tokens_per_page
                remaining_length = len(tokens) - n_cached_tokens
                n_empty_pages = math.ceil(remaining_length / self.tokens_per_page)
            else:
                n_empty_pages = math.ceil(len(tokens) / self.tokens_per_page)

            if not cached_pages and allocation_block_size > 0:
                n_empty_pages = allocation_block_size

            new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

            if new_pages is None and evict:
                # Try eviction
                self.evict_pages(n_empty_pages - len(self.page_pool.available_pages))
                new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

                if new_pages is None:
                    raise CacheAllocationFailure(
                        "Failed to acquire pages even after attempting eviction from LRU leaves"
                    )

            cur_node.ref_count.increment()
            pages = cached_pages + new_pages
            self._allocated_pages.extend(new_pages)

            num_tokens = len(tokens)
            if cache_info:
                if (
                    cache_info.last_cached_node
                    and not cache_info.last_cached_node.ref_count.is_empty()
                ):
                    cache_info.last_cached_node.ref_count.decrement()
                pages = cache_info.pages + pages
                num_tokens += cache_info.num_tokens

            return TrieCacheInfo(
                num_tokens=len(tokens),
                tokens=tokens,
                pages=pages,
                last_cached_node=cur_node,
                number_of_published_pages=len(cached_pages),
                pool=self.page_pool,
            )

    def extend_allocation(
        self, tokens: List[int], cache_info: TrieCacheInfo, *, extra_token_slots=0
    ) -> TrieCacheInfo:
        """Extend the current allocation to accommodate additional tokens.

        Args:
            tokens: New token sequence to extend the allocation to
            extra_token_slots: Additional token slots to allocate.
                - This allows us to allocate additional space for future token(s).

        Raises:
            ValueError: If new tokens don't extend current allocation's tokens
        """
        # Verify new tokens extend current tokens
        if len(tokens) < len(cache_info.tokens):
            raise ValueError("New tokens must be longer than current tokens")

        # Check that current tokens are a prefix of new tokens
        if tokens[: len(cache_info.tokens)] != cache_info.tokens:
            raise ValueError("New tokens must extend current token sequence")

        # If tokens are identical, no extension needed
        if len(tokens) == len(cache_info.tokens):
            return cache_info

        # Calculate how many new pages we need
        tokens_per_page = self.tokens_per_page
        current_pages = len(cache_info.pages)
        total_tokens = len(tokens) + extra_token_slots
        total_pages_needed = math.ceil(total_tokens / tokens_per_page)
        new_pages_needed = total_pages_needed - current_pages

        pages = cache_info.pages
        if new_pages_needed > 0:
            # Acquire new pages
            new_pages = self.page_pool.acquire_free_pages(new_pages_needed)

            if new_pages is None:
                # Try eviction if initial allocation fails
                self.evict_pages(new_pages_needed - len(self.page_pool.available_pages))
                new_pages = self.page_pool.acquire_free_pages(new_pages_needed)

                if new_pages is None:
                    raise CacheAllocationFailure(
                        "Failed to acquire pages for allocation extension even after attempting eviction"
                    )

            # Extend our page list
            pages.extend(new_pages)
        return TrieCacheInfo(
            num_tokens=len(tokens),
            tokens=deepcopy(tokens),
            pages=cache_info.pages,
            pool=cache_info.page_pool,
            last_cached_node=cache_info.last_cached_node,
            number_of_published_pages=cache_info.number_of_pages_to_publish,
        )

    def publish_pages_for_tokens(
        self, tokens: List[int], cache_info: TrieCacheInfo
    ) -> TrieCacheInfo:
        """Make pages available in the cache for the specified tokens.

        Args:
            tokens_to_publish: Tokens to publish to the cache
            cache_info: TrieCacheInfo object containing allocation metadata

        Raises:
            ValueError: If tokens don't match allocation or exceed available pages
        """
        with self._lock:
            # If we have more tokens, publish pages up to the incoming tokens.
            # If incoming has more tokens, replace our tokens with incoming tokens and publish pages up to the incoming tokens.
            updated_tokens = deepcopy(cache_info.tokens)
            tokens_per_page = self.tokens_per_page
            matched_node, matched_pages = self.match(updated_tokens)
            last_number_of_published_pages = cache_info.number_of_published_pages
            if len(matched_pages) > last_number_of_published_pages:
                last_number_of_published_pages = len(matched_pages)

            number_of_pages_to_publish = -(
                len(updated_tokens) // -tokens_per_page
            )  # ceil division

            # Create token blocks for unpublished pages
            start_token_index = last_number_of_published_pages * tokens_per_page
            unpublished_tokens = []

            unpublished_tokens.extend(
                [
                    tuple(updated_tokens[i : i + tokens_per_page])
                    for i in range(
                        start_token_index, len(updated_tokens), tokens_per_page
                    )
                ]
            )

            unpublished_pages = cache_info.pages[
                last_number_of_published_pages:number_of_pages_to_publish
            ]

            number_of_published_pages = 0

            cur_node = matched_node
            for token_block, page in zip(unpublished_tokens, unpublished_pages):
                new_node = cur_node.create_child(token_block, page)
                if page in self._allocated_pages:
                    self._allocated_pages.remove(page)

                # remove parent node from the leaves.
                # No need to delete if it was deleted earlier.
                if cur_node in self.leaves:
                    self.leaves.remove(cur_node)
                cur_node = new_node

                if cur_node is not self.root and cur_node not in self.leaves:
                    self.leaves.add(cur_node)

                if len(token_block) == tokens_per_page:
                    number_of_published_pages += 1

            # Update reference counts
            last_cached_node = cache_info.last_cached_node
            if unpublished_tokens:
                cur_node.ref_count.increment()
                if not last_cached_node.ref_count.is_empty():
                    last_cached_node.ref_count.decrement()
                last_cached_node = cur_node

            return TrieCacheInfo(
                num_tokens=len(updated_tokens),
                tokens=updated_tokens,
                pages=cache_info.pages,
                last_cached_node=last_cached_node,
                number_of_published_pages=number_of_published_pages,
                pool=self.page_pool,
            )

    def free_cache_pages(self):

        """Free all pages that have zero references."""

        pages_to_free = []
        # Initialize heap with unreferenced leaves
        unused_leaf_heap = [
            (leaf.access_time, leaf)
            for leaf in self.leaves
            if leaf.ref_count.is_empty()
        ]

        # Evict least recently used nodes
        while unused_leaf_heap:
            _, leaf = heapq.heappop(unused_leaf_heap)
            pages_to_free.append(leaf.page)
            parent = leaf.parent

            leaf.unlink()
            self.leaves.remove(leaf)

            # If parent becomes childless, it becomes a leaf
            if (
                parent is not self.root
                and not parent.children
                and parent not in self.leaves
            ):
                self.leaves.add(parent)
                if parent.ref_count.is_empty():
                    heapq.heappush(unused_leaf_heap, (parent.access_time, parent))

        if pages_to_free:
            self.page_pool.free_pages(pages_to_free)

        self.page_pool.free_pages(self._allocated_pages)

    def release_pages(self, cache_info: TrieCacheInfo):
        """Release the allocation's reference to its pages.

        Decrements reference count of the last cached node. When count
        reaches zero, the node becomes eligible for eviction.
        """
        if cache_info is None:
            return
        last_cached_node = cache_info.last_cached_node
        if not last_cached_node.ref_count.is_empty():
            last_cached_node.ref_count.decrement()

        self.page_pool.free_pages(self._allocated_pages)
        self._allocated_pages = []

    def shutdown(self):
        self.free_cache_pages()

        available = self.page_pool.available_page_count()
        total = self.page_pool.total_page_count()
        if available != total:
            raise ValueError(f"Pages lost: {total - available} of {total} unfreed")
