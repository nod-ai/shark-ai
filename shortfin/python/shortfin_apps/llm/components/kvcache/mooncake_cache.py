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
import shortfin as sf

import logging

logger = logging.getLogger(__name__)


def token_ids_to_key(token_ids: List[int]) -> str:
    """Convert a list of token ids to a unique key string.
    Args:
        token_ids: List of token ids
    Returns:
        A string key representing the token ids
    """
    key = "_".join(map(str, token_ids))
    return key


class MooncakePagedAllocation(PageAllocation):
    """Allocation for Mooncake store.

    Attributes:
        allocation: PageAllocation object that contains the allocation information
    """

    def __init__(self, allocation: PageAllocation, cache: BasePagedAttentionCache):
        self._allocation = allocation
        self._cache = cache

    @property
    def pages(self) -> List[PageInfo]:
        """Return the pages allocated for this allocation."""
        return self._allocation.pages

    @property
    def cache(self) -> BasePagedAttentionCache:
        """Return the cache associated with this allocation."""
        return self._cache

    def publish_pages_for_tokens(
        self, tokens, *, publish_incomplete_page=False
    ) -> None:
        self._allocation.publish_pages_for_tokens(
            tokens, publish_incomplete_page=publish_incomplete_page
        )

    def release_pages(self) -> None:
        """Release the pages allocated for this allocation."""
        self._allocation.release_pages()

    def extend_allocation(self, tokens, *, extra_token_slots=0) -> None:
        """Extend the allocation with additional tokens."""
        try:
            self._allocation.extend_allocation(
                tokens, extra_token_slots=extra_token_slots
            )
        except CacheAllocationFailure as e:
            logger.error(f"Cache allocation failed during extension: {e}")
            raise e

    def __rerp__(self) -> str:
        """Return a string representation of the MooncakePagedAllocation."""
        return f"MooncakePagedAllocation(pages={self.pages}, cache={self.cache})"

    async def write_back_pages(
        self, device: sf.ScopedDevice, token_ids: List[int]
    ) -> None:
        """read pages from device and send them to the Mooncake store."""
        page_pool = self.cache.page_pool
        tokens_per_page = self.cache.tokens_per_page
        number_of_pages = math.ceil(len(token_ids) / tokens_per_page)
        # copy pages from the device
        device_id = device.raw_device.node_affinity
        logger.debug(
            f"Write_back_pages for Device ID: {device_id}, number of tokens: {len(token_ids)}, Number of pages: {number_of_pages}, Tokens per page: {tokens_per_page}"
        )
        keys = []
        values = []

        for i in range(number_of_pages):
            start_index = i * tokens_per_page
            end_index = min((i + 1) * tokens_per_page, len(token_ids))
            page_tokens = token_ids[start_index:end_index]
            if not page_tokens:
                logger.warning(f"Skipping empty page for tokens: {page_tokens}")
                continue
            page_info = self.pages[i]
            # Get the page from the page pool
            value = page_pool.transfer_page_to_host(device_id=device_id, page=page_info)
            if not value:
                logger.error(f"Failed to get page for tokens: {page_tokens}")
                continue

            # Convert token ids to key
            key = token_ids_to_key(page_tokens)
            keys.append(key)
            values.append(value)

        await device

        mooncake_store = self.cache.mooncake_store
        for key, value in zip(keys, values):
            logger.debug(f"Writing back page to Mooncake store with key: {key}")
            mooncake_store.put_int_list(key, value)
        logger.info(f"Successfully wrote back {len(keys)} pages to Mooncake store.")

    async def update_pages(self, device: sf.ScopedDevice, token_ids: List[int]) -> bool:
        """Update pages in the device.
        This method splits the token_ids into keys according to tokens_per_page used in page_pool, retrieves the page correspons to the key from Mooncake and updates corresponding page in the page pool.
        args:
            device: Device to update pages on
            token_ids: List of token ids to update pages for
        returns:
            True if all pages were updated successfully, False otherwise
        """
        page_pool = self.cache.page_pool
        tokens_per_page = self.cache.tokens_per_page
        number_of_pages = math.ceil(len(token_ids) / tokens_per_page)
        device_id = device.raw_device.node_affinity
        logger.debug(
            f"Update_pages for Device ID: {device_id}, number of tokens: {len(token_ids)}, Number of pages: {number_of_pages}, Tokens per page: {tokens_per_page}"
        )

        mooncake_store = self.cache.mooncake_store
        values = []
        for i in range(number_of_pages):
            start_index = i * tokens_per_page
            end_index = min((i + 1) * tokens_per_page, len(token_ids))
            page_tokens = token_ids[start_index:end_index]
            if not page_tokens:
                logger.warning(f"Skipping empty page for tokens: {page_tokens}")
                continue

            key = token_ids_to_key(page_tokens)
            value = mooncake_store.get_int_list(key)
            if value is None:
                logger.warning(
                    f"Page not found in Mooncake store for key: {key}. Skipping updating this and the rest of tokens."
                )
                continue
            logger.debug(f"Got page for key: {key} from Mooncake store")
            # Get the page from the page pool
            values.append(value)

        for i in range(len(values)):
            page_info = self.pages[i]
            # Update the page in the page pool
            page_pool.update_device_page(
                device_id=device_id,
                page=page_info,
                data=values[i],
            )
        await device
        logger.info(f"Updated {len(values)} pages in the device.")
        if len(values) == number_of_pages:
            return True
        else:
            logger.warning(
                f"Only updated {len(values)} out of {number_of_pages} pages in the device."
            )
            return False


class MooncakeAttentionCache(BasePagedAttentionCache):
    """Distributed KVcache implemented with Mooncake store.

    refresh and write back the page cache from / to Mooncake store.

    Attributes:
        mooncake_store: mooncake store client for persistent storage
        page_cache: page cache for local KV pair storage
    """

    def __init__(
        self,
        page_pool: PagePool,
        tokens_per_page: int,
        prefix_sharing_algorithm: str,
        mooncake_config_path: str,
    ):
        """Initialize the mooncacke cache.

        Args:
            mooncake_config_path: Path to Mooncake configuration file
            prefix_sharing_algorithm: Algorithm to use for prefix sharing
            page_pool: Pool to allocate pages from
            tokens_per_page: Number of tokens per page
        """
        if prefix_sharing_algorithm == "trie":
            self.page_cache = TriePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=tokens_per_page,
            )
        else:
            self.page_cache = BasePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=tokens_per_page,
            )

        self.mooncake_config_path = mooncake_config_path
        mooncake_config = MooncakeConfig.from_json(self.mooncake_config_path)
        self.mooncake_store = MooncakeStore(mooncake_config)
        self.mooncake_keys: Set[str] = set()
        logger.info(f"Mooncake store enabled with config: {self.mooncake_config_path}")

    @property
    def page_pool(self) -> PagePool:
        """Return the page pool associated with this cache."""
        return self.page_cache.page_pool

    @property
    def tokens_per_page(self) -> int:
        """Return the number of tokens per page."""
        return self.page_cache.tokens_per_page

    @property
    def use_ref_counts(self) -> bool:
        """Return whether to use reference counts for pages."""
        return self.page_cache.use_ref_counts

    def acquire_pages_for_tokens(
        self, tokens: List[int], extra_token_slots: int = 1
    ) -> PageAllocation:
        """Acquire pages for the given tokens.

        Args:
            tokens: List of token ids to acquire pages for
            extra_token_slots: Number of extra token slots to allocate

        Returns:
            PageAllocation object containing the allocated pages
        """
        try:
            allocation = self.page_cache.acquire_pages_for_tokens(
                tokens, extra_token_slots=extra_token_slots
            )
        except CacheAllocationFailure as e:
            logger.error(f"Cache allocation failed: {e}")
            raise e
        logger.debug(f"type of acquired allocation: {type(allocation)}")
        return MooncakePagedAllocation(allocation, self)

    def increment_pages(self, pages: List[PageInfo]):
        self.page_cache.increment_pages(pages)

    def decrement_pages(
        self, pages: List[PageInfo], return_empty_pages: bool = False
    ) -> None | List[PageInfo]:
        self.page_cache.decrement_pages(pages, return_empty_pages=return_empty_pages)

    def free_pages(self, pages: List[PageInfo]):
        self.page_cache.free_pages(pages)

    def fork_pages(self, pages: List[PageInfo]) -> List[PageInfo]:
        return self.page_cache.fork_pages(pages)
