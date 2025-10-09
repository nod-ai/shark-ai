from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from threading import Lock
import time
import math
import heapq
from copy import deepcopy
from .page_pool import PagePool, PageInfo
from .base_attention_cache import BasePagedAttentionCache, CacheAllocationFailure
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
        new_node = None
        if tokens in self.children:
            # If the child already exists, return it
            new_node = self.children[tokens]
        else:
            new_node = TrieNode(tokens=tokens, page=page, parent=self)
            self.children[tokens] = new_node
        return new_node

    def register_allocation(self):
        """Increment the reference count for this node to register that more following pages have been allocated."""
        self.ref_count.increment()

    def publish_descendant(self, descendant: "TrieNode") -> None:
        """Because ref_count is used to track allocations that depend on this node, if we have already created descendant nodes for the allocation, we need to decrease the ref_count of this node and increase the ref_count of the descendant node to reflect that the allocations have already been recorded as the descends of this node, and the new allocation depends on the descendant node."""
        if not self.ref_count.is_empty():
            self.ref_count.decrement()
        descendant.ref_count.increment()

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
        last_cached_node: Last node in the trie that was cached
        number_of_published_pages: Number of pages that have been published to the cache
    """

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

        super().__init__(page_pool, tokens_per_page)

        # Create root node with dummy page
        dummy_page = PageInfo(
            index=0,  # Root uses reserved index 0
            pool=self.page_pool,
        )
        self.root = TrieNode(tokens=tuple(), page=dummy_page)
        self.leaves: Set[TrieNode] = set()
        self._lock: Lock = Lock()
        self._duplicated_pages: List[
            PageInfo
        ] = (
            []
        )  # pages that are duplicated from existing pages in Trie tree. These pages can be safely freed when calling release_pages.

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

    def lookup(self, tokens: List[int]) -> TrieCacheInfo:
        """Lookup the cache for the given token sequence. It only returns fully matched pages.

        Args:
            tokens: Sequence of tokens to look up
            returns: TrieCacheInfo with matched tokens and pages
        """
        with self._lock:
            page_aligned_token_len = (
                len(tokens) // self.tokens_per_page
            ) * self.tokens_per_page
            page_aligned_tokens = tokens[:page_aligned_token_len]
            cur_node, matched_pages = self.match(page_aligned_tokens)
            num_matched_tokens = len(matched_pages) * self.tokens_per_page
            matched_tokens = page_aligned_tokens[:num_matched_tokens]
            return TrieCacheInfo(
                num_tokens=num_matched_tokens,
                tokens=matched_tokens,
                pages=matched_pages,
                last_cached_node=cur_node,
                number_of_published_pages=len(matched_pages),
                pool=self.page_pool,
            )

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
            if leaf.page in pages_to_evict:
                continue
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
            self.page_pool.free_pages(pages_to_evict)

        return len(pages_to_evict)

    def allocate(
        self,
        tokens: List[int],
        cache_info: TrieCacheInfo = None,
        allocation_block_size: int = 0,
        evict: bool = True,
    ) -> TrieCacheInfo:
        """Acquire pages for a sequence of tokens.

        Attempts to reuse existing cached pages where possible through
        prefix matching, allocating new pages only for the uncached suffix.

        Args:
            tokens: Sequence of tokens needing pages
            cache_info: Existing TrieCacheInfo to extend/update, if any
            allocation_block_size: number of pages to allocate at once, not used if it is 0
            evict: Whether to evict old tokens if the cache is full.

        Returns:
            PageAllocation containing both cached and newly allocated pages

        Raises:
            CacheAllocationFailure: If unable to allocate required pages
        """
        with self._lock:
            tokens = tuple(tokens)
            n_empty_pages = 0
            pages = []
            cur_node = self.root
            if not cache_info:
                raise ValueError("cache_info cannot be None")

            cur_node = cache_info.last_cached_node

            n_empty_pages = math.ceil(len(tokens) / self.tokens_per_page)

            if allocation_block_size > 0:
                n_empty_pages = max(n_empty_pages, allocation_block_size)

            new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

            if new_pages is None and evict:
                # Try eviction
                number_evicted_pages = self.evict_pages(
                    n_empty_pages - len(self.page_pool.available_pages)
                )
                new_pages = self.page_pool.acquire_free_pages(n_empty_pages)

                if new_pages is None:
                    raise CacheAllocationFailure(
                        "Failed to acquire pages even after attempting eviction from LRU leaves"
                    )

            if new_pages is None:
                raise CacheAllocationFailure(
                    "Failed to acquire pages and eviction is disabled"
                )

            if len(new_pages) > 0:
                # some new pages are allocated and will be used to create children of cur_node, hence increment ref_count of cur_node.
                # do not increment last_cached_node ref_count when we allocate along the same branch again
                if len(cache_info.pages) == 0 or (
                    len(cache_info.pages) > 0
                    and cur_node.page.index == cache_info.pages[-1].index
                ):
                    cur_node.register_allocation()

            self._allocated_pages.extend(new_pages)
            pages = cache_info.pages + new_pages
            num_tokens = len(tokens) + cache_info.num_tokens
            tokens = cache_info.tokens + list(tokens)
            number_of_published_pages = cache_info.number_of_published_pages

            return TrieCacheInfo(
                num_tokens=num_tokens,
                tokens=tokens,
                pages=pages,
                last_cached_node=cur_node,
                number_of_published_pages=number_of_published_pages,
                pool=self.page_pool,
            )

    def publish_pages_for_tokens(
        self, cache_info: TrieCacheInfo, publish_incomplete_page: bool = False
    ) -> TrieCacheInfo:
        """Make pages available in the cache for the specified tokens.

        Args:
            cache_info: TrieCacheInfo object containing allocation metadata
            publish_incomplete_page: Whether to publish the last page even if it is not full

        Raises:
            ValueError: If tokens don't match allocation or exceed available pages
        """
        with self._lock:
            # If we have more tokens, publish pages up to the incoming tokens.
            # If incoming has more tokens, replace our tokens with incoming tokens and publish pages up to the incoming tokens.
            if not cache_info:
                raise ValueError("cache_info cannot be None")
            updated_tokens = deepcopy(cache_info.tokens)
            tokens_per_page = self.tokens_per_page
            number_of_pages_to_publish = len(updated_tokens) // tokens_per_page
            matched_node, matched_pages = self.match(
                updated_tokens[: number_of_pages_to_publish * tokens_per_page]
            )

            duplicated_page_set = set([page.index for page in self._duplicated_pages])
            for i, page in enumerate(matched_pages):
                if page.index != cache_info.pages[i].index:
                    if page.index not in duplicated_page_set:
                        self._duplicated_pages.append(cache_info.pages[i])
                    if cache_info.last_cached_node.page.index == page.index:
                        cache_info.last_cached_node.page = page

            last_number_of_published_pages = cache_info.number_of_published_pages

            if len(matched_pages) > last_number_of_published_pages:
                last_number_of_published_pages = len(matched_pages)

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
                last_number_of_published_pages : last_number_of_published_pages
                + len(unpublished_tokens)
            ]
            number_of_published_pages = last_number_of_published_pages

            pages = matched_pages  # using matched pages instead of cache_info.pages to avoid using the _duplicated pages

            last_cached_node = cache_info.last_cached_node
            cur_node = matched_node
            for token_block, page in zip(
                unpublished_tokens[: len(unpublished_pages)], unpublished_pages
            ):
                if not publish_incomplete_page and len(token_block) < tokens_per_page:
                    # Do not publish incomplete page
                    break
                new_node = cur_node.create_child(token_block, page)
                if new_node.page.index != page.index:
                    if page not in self._duplicated_pages:
                        self._duplicated_pages.append(page)
                pages.append(new_node.page)
                # remove parent node from the leaves.
                # No need to delete if it was deleted earlier.
                if cur_node in self.leaves:
                    self.leaves.remove(cur_node)
                cur_node = new_node

                if cur_node is not self.root and cur_node not in self.leaves:
                    self.leaves.add(cur_node)

                # we create a new node for each token block, but we only publish full pages, hence last_cached_node is updated only when a full page is published.
                if len(token_block) == tokens_per_page:
                    number_of_published_pages += 1
                    last_cached_node = new_node

            # Update reference counts only when we have unpublished tokens
            if unpublished_tokens:
                cache_info.last_cached_node.publish_descendant(last_cached_node)

            # Remove published pages from _allocated_pages
            for page in pages:
                if page in self._allocated_pages:
                    self._allocated_pages.remove(page)
            # if we don't publish the last incomplete page, and len(pages) < len(cache_info.pages), we should return cache_info.pages to avoid losing the reference to the last incomplete page.
            if not publish_incomplete_page and len(pages) < len(cache_info.pages):
                pages = pages + cache_info.pages[len(pages) :]

            return TrieCacheInfo(
                num_tokens=len(updated_tokens),
                tokens=updated_tokens,
                pages=pages,
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
        self._allocated_pages = []

    def free_allocated_pages(self, page_ids: List[int]):
        page_id_set = set(page_ids)
        pages = [page for page in self._allocated_pages if page.index in page_id_set]
        for page in pages:
            self._allocated_pages.remove(page)
        self.page_pool.free_pages(pages)

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

        # free duplicated pages
        self.page_pool.free_pages(self._duplicated_pages)
        self._duplicated_pages = []

    def shutdown(self):
        self.free_cache_pages()

        available = self.page_pool.available_page_count()
        total = self.page_pool.total_page_count()

        if available != total:
            raise ValueError(f"Pages lost: {total - available} of {total} unfreed")
