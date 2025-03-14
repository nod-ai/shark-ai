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


@dataclass
class RefCount:
    """
    A reference counter to replace simple int.
    """

    count: int = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

    def decrement(self) -> int:
        self.count -= 1
        return self.count

    def is_empty(self) -> bool:
        return self.count <= 0


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


class TriePagedAttentionCacheAllocation(PageAllocation):
    """Represents a page allocation in the trie-based cache.

    Tracks sequence of pages and which ones are already published to the cache,
    implementing the PageAllocation protocol for the trie cache.

    Attributes:
        cache: The parent cache this allocation belongs to
        tokens: Complete sequence of tokens this allocation represents
        last_cached_node: Last matched node in the trie
        pages: List of all pages in allocation
        number_of_published_pages: Number of pages that are published to the cache
    """

    def __init__(
        self,
        cache: "TriePagedAttentionCache",
        tokens: List[int],
        last_cached_node: TrieNode,
        cached_pages: List[PageInfo],
        newly_acquired_pages: List[PageInfo],
    ):
        self.cache = cache
        self.tokens = tokens
        self.last_cached_node = last_cached_node
        self._pages = cached_pages + newly_acquired_pages
        self.number_of_published_pages = len(cached_pages)
        self._is_released = False

    @property
    def pages(self) -> List[PageInfo]:
        return self._pages

    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        """Make pages available in the cache for the specified tokens.

        Args:
            tokens_to_publish: Tokens to publish to the cache

        Raises:
            ValueError: If tokens don't match allocation or exceed available pages
        """
        # If we have more tokens, publish pages up to the incoming tokens.
        # If incoming has more tokens, replace our tokens with incoming tokens and publish pages up to the incoming tokens.

        def has_common_prefix(tokens1, tokens2):
            for t1, t2 in zip(tokens1, tokens2):
                if t1 != t2:
                    return False
            return True

        if not has_common_prefix(self.tokens, tokens):
            raise ValueError(
                "Tokens provided in publish_pages do not match tokens in allocation"
            )

        if len(tokens) > len(self.tokens):
            self.tokens = tokens

        tokens_per_page = self.cache.tokens_per_page

        if publish_incomplete_page:
            number_of_pages_to_publish = -(
                len(tokens) // -tokens_per_page
            )  # ceil division
        else:
            number_of_pages_to_publish = len(tokens) // tokens_per_page

        # Create token blocks for unpublished pages
        start_token_index = self.number_of_published_pages * tokens_per_page
        unpublished_tokens = [
            tuple(self.tokens[i : i + tokens_per_page])
            for i in range(start_token_index, len(self.tokens), tokens_per_page)
        ]

        unpublished_pages = self._pages[
            self.number_of_published_pages : number_of_pages_to_publish
        ]

        # Add unpublished pages to trie
        if publish_incomplete_page:
            raise NotImplementedError(
                "Additional work needed here to support publishing incomplete pages to ensure that we finish up a page before attaching child nodes to it."
            )
        cur_node = self.last_cached_node
        for token_block, page in zip(unpublished_tokens, unpublished_pages):
            new_node = cur_node.create_child(token_block, page)
            cur_node = new_node

        if cur_node is not self.cache.root:
            self.cache.leaves.add(cur_node)

        # Update reference counts
        if unpublished_tokens:
            cur_node.ref_count.increment()
            self.last_cached_node.ref_count.decrement()
            self.last_cached_node = cur_node

        self.number_of_published_pages = number_of_pages_to_publish

    def release_pages(self) -> None:
        """Release the allocation's reference to its pages.

        Decrements reference count of the last cached node. When count
        reaches zero, the node becomes eligible for eviction.
        """
        if self._is_released:
            return

        self.last_cached_node.ref_count.decrement()
        self._is_released = True

    def extend_allocation(self, tokens: List[int], *, extra_token_slots=0) -> None:
        """Extend the current allocation to accommodate additional tokens.

        Args:
            tokens: New token sequence to extend the allocation to

        Raises:
            ValueError: If new tokens don't extend current allocation's tokens
        """
        # Verify new tokens extend current tokens
        if len(tokens) < len(self.tokens):
            raise ValueError("New tokens must be longer than current tokens")

        # Check that current tokens are a prefix of new tokens
        for old_token, new_token in zip(self.tokens, tokens):
            if old_token != new_token:
                raise ValueError("New tokens must extend current token sequence")

        # If tokens are identical, no extension needed
        if len(tokens) == len(self.tokens):
            return

        # Calculate how many new pages we need
        tokens_per_page = self.cache.tokens_per_page
        current_pages = len(self._pages)
        total_tokens = len(tokens) + extra_token_slots
        total_pages_needed = math.ceil(total_tokens / tokens_per_page)
        new_pages_needed = total_pages_needed - current_pages

        if new_pages_needed <= 0:
            self.tokens = tokens
            return

        # Acquire new pages
        new_pages = self.cache.page_pool.acquire_free_pages(new_pages_needed)

        if new_pages is None:
            # Try eviction if initial allocation fails
            self.cache._evict_pages(
                new_pages_needed - len(self.cache.page_pool.available_pages)
            )
            new_pages = self.cache.page_pool.acquire_free_pages(new_pages_needed)

            if new_pages is None:
                raise CacheAllocationFailure(
                    "Failed to acquire pages for allocation extension even after attempting eviction"
                )

        # Extend our page list
        self._pages.extend(new_pages)

        # Update tokens
        self.tokens = tokens


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

    def _match(self, tokens: List[int]) -> Tuple[TrieNode, List[PageInfo]]:
        """
        Find the longest prefix match in the trie.

        Walks the trie following the token sequence as far as possible,
        collecting matched pages along the way.

        Args:
            tokens: Sequence of tokens to match

        Returns:
            Tuple of (last matched node, list of matched pages)
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
            return TriePagedAttentionCacheAllocation(
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

        return TriePagedAttentionCacheAllocation(
            cache=self,
            tokens=tokens,
            last_cached_node=cur_node,
            cached_pages=matched_pages,
            newly_acquired_pages=new_pages,
        )

    def _evict_pages(self, max_pages: int) -> int:
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
            if parent is not self.root and not parent.children:
                self.leaves.add(parent)
                if parent.ref_count.is_empty():
                    heapq.heappush(unused_leaf_heap, (parent.access_time, parent))

        if pages_to_evict:
            self.page_pool.free_pages(pages_to_evict)

        return len(pages_to_evict)
