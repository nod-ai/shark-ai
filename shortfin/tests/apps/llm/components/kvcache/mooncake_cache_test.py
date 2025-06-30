"""
Test mooncake_cake with a mocked MooncakeStore
"""

import asyncio
import pytest
import threading
import queue
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional
import shortfin as sf
import shortfin.array as sfnp
from concurrent.futures import ThreadPoolExecutor

import logging

from shortfin_apps.llm.components.kvcache.mooncake_cache import (
    MooncakeAttentionCache,
)
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    CacheAllocationFailure,
)
from shortfin_apps.llm.components.kvcache.page_pool import (
    PagePool,
    PagePoolConfig,
)

logger = logging.getLogger(__name__)

# Test constants
TEST_PAGE_SIZE = 16  # Tokens per page

# Using small block size and pool capacity for testing
TEST_BLOCK_SIZE = 8
TEST_POOL_CAPACITY = 256


@pytest.fixture
def mooncake_store():
    """
    Mocked MooncakeStore fixture
    """

    class MockMooncakeStore:
        def __init__(self):
            self.store = {}

        def put_int_list(self, key: str, value: list[int]) -> None:
            self.store[key] = value

        def get_int_list(self, key: str) -> Optional[list[int]]:
            if key in self.store:
                return self.store[key]
            return None

        def close(self):
            self.store.clear()

    return MockMooncakeStore()


@pytest.fixture
def real_device():
    """Create a real device using the system manager"""
    sc = sf.host.CPUSystemBuilder()
    with sc.create_system() as ls:
        worker = ls.create_worker("test-worker")
        fiber = ls.create_fiber(worker)
        yield list(fiber.devices_dict.values())[0]  # Get the first device


@pytest.fixture
def page_pool(real_device):
    """Create a real PagePool with test parameters"""
    config = PagePoolConfig(
        dtype=sfnp.float32,  # Using float32 as requested
        alloc_page_count=TEST_POOL_CAPACITY,  # Using 256 pages as requested
        paged_kv_block_size_elements=TEST_BLOCK_SIZE,  # Using small block size (8) for testing
    )

    return PagePool(devices=[real_device], config=config)


@pytest.fixture
def mooncake_cache(page_pool, mooncake_store):
    """
    Create a MooncakeAttentionCache instance with the mocked MooncakeStore
    """
    return MooncakeAttentionCache(
        page_pool=page_pool,
        tokens_per_page=TEST_PAGE_SIZE,
        prefix_sharing_algorithm="none",
        mooncake_config_path="",  # Not used in this test
        mooncake_store=mooncake_store,
    )


test_data = [
    # Tokens                                keys  Case Name
    ([], 0, "empty_token_list"),
    (list(range(TEST_PAGE_SIZE // 2)), 1, "partial_page"),
    (list(range(TEST_PAGE_SIZE)), 1, "exact_page"),
    (list(range(TEST_PAGE_SIZE + 1)), 2, "just_over_one_page"),
    (list(range(TEST_PAGE_SIZE * 2)), 2, "multiple_exact_pages"),
    (list(range(TEST_PAGE_SIZE * 2 + 1)), 3, "multiple_pages_with_rema"),
    (list(range(TEST_PAGE_SIZE * 3)), 3, "three_exact_pages"),
    (list(range(1)), 1, "single_token"),
    (list(range(TEST_PAGE_SIZE - 1)), 1, "almost_full_page"),
]

# fmt: off
@pytest.mark.parametrize(
   "tokens,expected_keys,case_name", test_data
)
# fmt: on
def test_write_back_pages(
    mooncake_cache, real_device, tokens, expected_keys, case_name
):
    async def main():
        allocation = mooncake_cache.acquire_pages_for_tokens(tokens)
        num_stored_kvs = await allocation.write_back_pages(real_device, tokens)
        assert num_stored_kvs == expected_pages, f"Failed case: {case_name}"
        allocation.release_pages()


# fmt: off
@pytest.mark.parametrize(
   "tokens,expected_keys,case_name", test_data
)
def test_update_pages(
    mooncake_cache, real_device, tokens, expected_keys, case_name
):
    async def main():
        allocation = mooncake_cache.acquire_pages_for_tokens(tokens)
        num_stored_kvs = await allocation.write_back_pages(real_device, tokens)
        laste_written_values = allocation.last_written_back_values
        num_updated_kvs = await allocation.update_pages(real_device, tokens)
        last_updated_values = allocation.last_updated_values
        assert num_stored_kvs == expected_keys, f"Failed case: {case_name}"
        assert num_updated_kvs == expected_keys, f"Failed case: {case_name}"
        assert(laste_written_values == last_updated_values), \
            f"Last written values do not match last updated values in case: {case_name}"
        allocation.release_pages()
