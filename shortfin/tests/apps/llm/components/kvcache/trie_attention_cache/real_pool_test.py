"""
Trie attention cache tests with a real page pool.

This file contains tests that involve writing data to the page. Tests that deal purely with trie cache structure should go in `mock_pool_tests.py`.

Each test requires us to initialize a new page pool & page table device array. Tests here will be a LOT slower.
"""


import pytest
from typing import List
import shortfin as sf
import shortfin.array as sfnp
import time
import logging
from dataclasses import dataclass

from shortfin_apps.llm.components.kvcache.trie_attention_cache import (
    TriePagedAttentionCache,
)
from shortfin_apps.llm.components.kvcache.page_pool import (
    PagePool,
    PagePoolConfig,
)


# Test constants
TEST_PAGE_SIZE = 16  # Tokens per page

# Note: Using a very small block size (8 elements) for testing purposes.
# In real applications, this would typically be much larger for performance reasons.
TEST_BLOCK_SIZE = 8
TEST_POOL_CAPACITY = 256


# set up logging
logger = logging.getLogger(__name__)


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
def trie_cache(page_pool):
    """Create TriePagedAttentionCache instance"""
    return TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)


@pytest.fixture
def published_sequence(trie_cache):
    """Helper fixture that returns a function to publish token sequences"""

    def _publish_sequence(tokens: List[int]) -> None:
        alloc = trie_cache.acquire_pages_for_tokens(tokens, extra_token_slots=0)
        alloc.publish_pages_for_tokens(alloc.tokens)
        alloc.release_pages()

    return _publish_sequence


@pytest.mark.xfail(reason="Partial page reuse is not yet implemented.", strict=True)
def test_partial_page_publishing(trie_cache):
    """Test that we can publish partial pages and match them correctly"""
    # Create a sequence that's 1.5 pages long and publish it
    tokens = list(range(TEST_PAGE_SIZE + TEST_PAGE_SIZE // 2))
    alloc1 = trie_cache.acquire_pages_for_tokens(tokens)
    # write to the first page

    alloc1.publish_pages_for_tokens(tokens)

    # Try to match exactly half of the second page
    match_tokens = tokens[: TEST_PAGE_SIZE + TEST_PAGE_SIZE // 2]
    alloc2 = trie_cache.acquire_pages_for_tokens(match_tokens)

    # We should match both the full first page and half of the second page
    assert (
        alloc2.number_of_published_pages == 2
    ), "Should match both pages, including the partial one"
    # We should not get the same second page
    assert (
        alloc2.pages[1].index != alloc1.pages[1].index
    ), "Should not match the same second page"
