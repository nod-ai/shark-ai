import pytest
import logging
from shortfin_apps.llm.components.kvcache.page_pool import (
    PagePool,
    PagePoolConfig,
    RefCount,
)
import shortfin.array as sfnp

logger = logging.getLogger(__name__)


@pytest.fixture
def setup_pool(generic_device):
    pool = PagePool(
        devices=[generic_device],
        config=PagePoolConfig(
            alloc_page_count=256,
            dtype=sfnp.float16,
            paged_kv_block_size_elements=393216,
        ),
    )
    return pool


def test_page_acquisition(setup_pool):
    pool = setup_pool
    logger.info(f"=== Running page acquisition test on system ===")
    page0 = pool.acquire_free_pages(1)
    assert page0 is not None, f"Failed to acquire a free page on system"
    logger.info(f"Successfully acquired page on system")


def test_page_acquisition(setup_pool):
    pool = setup_pool
    logger.info(f"=== Running page acquisition test on system ===")
    page0 = pool.acquire_free_pages(1)
    assert page0 is not None, f"Failed to acquire a free page on system"
    assert not pool.is_available(page0[0])

    pool.free_pages(page0)
    assert pool.is_available(page0[0])
    logger.info(f"Successfully acquired page on system")


def test_page_retain(setup_pool):
    pool = setup_pool
    logger.info(f"=== Running page retain test on system ===")
    page0 = pool.acquire_free_pages(1)
    assert page0 is not None, f"Failed to acquire a free page on system"
    assert pool.retain_count(page0[0]) == 1
    assert not pool.is_available(page0[0])

    pool.retain_pages(page0)
    assert pool.retain_count(page0[0]) == 2
    assert not pool.is_available(page0[0])

    pool.free_pages(page0)
    assert pool.retain_count(page0[0]) == 1
    assert not pool.is_available(page0[0])

    pool.free_pages(page0)
    assert pool.retain_count(page0[0]) == 0
    assert pool.is_available(page0[0])
    logger.info(f"Successfully acquired page on system")


def test_page_copy(setup_pool):
    pool = setup_pool
    logger.info(f"=== Running page copy test on system ===")
    (page0,) = pool.acquire_free_pages(1)
    page1 = pool.copy_page(page0)
    assert page1 is not None, f"Failed to copy a page on system"
    assert page0 != page1, f"Copied page should be different from original on system"
    logger.info(f"Successfully copied page on system")


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging format to include timestamp and level"""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )


# Add more tests as needed

if __name__ == "__main__":
    pytest.main([__file__])
