# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import logging
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PagePoolConfig
import shortfin as sf
import shortfin.host
import shortfin.array as sfnp
import shortfin.amdgpu

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
