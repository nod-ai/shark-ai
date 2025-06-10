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
        self.mooncake_store = MooncakeStore(mooncake_config_path)
        print(f"Mooncake store enabled with config: {self.mooncake_config_path}")
