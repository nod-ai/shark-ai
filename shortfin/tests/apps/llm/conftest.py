# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import queue
from typing import List
import pytest

from shortfin.support.deps import ShortfinDepNotFoundError


@pytest.fixture(autouse=True)
def require_deps():
    try:
        import shortfin_apps.llm
    except ShortfinDepNotFoundError as e:
        pytest.skip(f"Dep not available: {e}")


import shortfin as sf
import shortfin.array as sfnp

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PageInfo


@pytest.fixture(scope="module")
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture(scope="module")
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture(scope="module")
def device(fiber):
    return fiber.device(0)


TEST_PAGE_SIZE = 16
TEST_POOL_CAPACITY = 10


class MockPagePool(PagePool):
    def __init__(self, total_pages: int, device):
        self._queue = queue.Queue()
        self.attn_page_entries = []
        for i in range(total_pages):
            page = PageInfo(index=i, pool=self)
            self._queue.put(page)
            self.attn_page_entries.append(page)

        self.page_tables = []

        # Set up a basic page table with shape [num_pages, 16].
        # Here, 16 is just an arbitrary value to denote the vocab size.
        page_table_shape = [total_pages, 16]
        page_table = sfnp.device_array.for_device(
            device,
            page_table_shape,
            dtype=sfnp.float32,
        )
        page_table_host = page_table.for_transfer()
        with page_table_host.map(discard=True) as m:
            m.fill(0)
        page_table_host.copy_to(page_table)
        self.page_tables.append(page_table)

    def acquire_free_pages(self, count: int) -> List[PageInfo]:
        try:
            return [self._queue.get_nowait() for _ in range(count)]
        except queue.Empty:
            return None

    def free_pages(self, pages):
        for page in pages:
            self._queue.put(page)


@pytest.fixture
def page_pool(device):
    return MockPagePool(total_pages=TEST_POOL_CAPACITY, device=device)


@pytest.fixture
def dummy_pages(page_pool) -> List[PageInfo]:
    return [PageInfo(index=i, pool=page_pool) for i in range(3)]


@pytest.fixture
def cache(page_pool):
    yield BasePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)


@pytest.fixture(scope="function")
def cache_ref_count(page_pool):
    yield BasePagedAttentionCache(
        page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE, use_ref_counts=True
    )
