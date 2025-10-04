# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import pytest
import threading
import queue
import random
import time
from collections import defaultdict
from typing import List

# import shortfin as sf
import shortfin.array as sfnp

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    CacheAllocationFailure,
)
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PageInfo


TEST_PAGE_SIZE = 16
TEST_POOL_CAPACITY = 10


# fmt: off
@pytest.mark.parametrize(
   "tokens,expected_pages,case_name",
   [   # Tokens                                Pages  Case Name
       ([],                                    0,     "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      1,     "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           1,     "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       2,     "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       2,     "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   3,     "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       3,     "three_exact_pages"),
       (list(range(1)),                        1,     "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       1,     "almost_full_page"),
   ],
)
# fmt: on
def test_allocation_sizes(cache, tokens, expected_pages, case_name):
    allocation = cache.allocate(tokens)
    pages = allocation.pages
    assert len(pages) == expected_pages, f"Failed for case: {case_name}"
    assert allocation.num_tokens == len(
        tokens
    ), f"Token count mismatch for case: {case_name}"
    cache.free_pages(pages)


# fmt: off
@pytest.mark.parametrize(
   "tokens,expected_pages,case_name",
   [   # Tokens                                Pages  Case Name
       ([],                                    0,     "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      1,     "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           1,     "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       2,     "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       2,     "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   3,     "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       3,     "three_exact_pages"),
       (list(range(1)),                        1,     "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       1,     "almost_full_page"),
   ],
)
# fmt: on
def test_allocation_ref_counts(cache_ref_count, tokens, expected_pages, case_name):
    allocation = cache_ref_count.allocate(tokens)
    pages = allocation.pages
    assert len(pages) == expected_pages, f"Allocation failed for case: {case_name}"


# fmt: off
@pytest.mark.parametrize(
  "num_workers,pages_per_worker,expect_failure,case_name",
  [   # Workers                 Pages   Failure  Case name
      (2,                       1,      False,  "basic_concurrent"),               # Basic concurrent access
      (5,                       1,      False,  "high_concurrency"),               # Higher concurrency, single page
      (3,                       2,      False,  "multi_page"),                     # Multiple pages per worker
      (2,                       3,      False,  "more_pages"),                     # More pages than workers, within capacity
      (TEST_POOL_CAPACITY,      1,      False,  "max_capacity"),                   # Max capacity single pages
      (TEST_POOL_CAPACITY // 2, 2,      False,  "max_capacity_multi"),             # Max capacity multiple pages
      (4,                       3,      True ,  "exceeds_total"),                  # 12 pages needed, exceeds capacity
      (TEST_POOL_CAPACITY + 1,  1,      True ,  "exceeds_workers"),                # More workers than capacity
      (TEST_POOL_CAPACITY // 2, 3,      True ,  "exceeds_with_multi"),             # Exceeds capacity with multiple pages
  ],
)
# fmt: on
def test_concurrent_page_allocation(
    cache,
    num_workers,
    pages_per_worker,
    expect_failure,
    case_name,
):
    allocated_pages = defaultdict(set)
    errors = []
    allocations = []

    def worker(worker_id: int):
        try:
            tokens = list(range(TEST_PAGE_SIZE * pages_per_worker))
            allocation = cache.allocate(tokens)
            allocations.append(allocation)
            allocated_pages[worker_id] = {page.index for page in allocation.pages}
            time.sleep(random.uniform(0.001, 0.01))
        except CacheAllocationFailure as e:
            errors.append(e)
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if expect_failure:
        assert len(errors) > 0, "Expected at least one CacheAllocationFailure"
    else:
        assert not errors, f"Workers encountered errors: {errors}"
        for worker_id, pages in allocated_pages.items():
            assert (
                len(pages) == pages_per_worker
            ), f"Worker {worker_id} got {len(pages)} pages, expected {pages_per_worker}"

        all_pages = set()
        for pages in allocated_pages.values():
            assert not (
                pages & all_pages
            ), f"Found duplicate page allocation: {pages & all_pages}"
            all_pages.update(pages)

    for allocation in allocations:
        cache.free_pages(allocation.pages)


# fmt: off
@pytest.mark.parametrize(
  "num_workers,pages_per_worker,expect_failure,case_name",
  [   # Workers                 Pages   Failure  Case name
      (2,                       1,      False,  "basic_concurrent"),               # Basic concurrent access
      (5,                       1,      False,  "high_concurrency"),               # Higher concurrency, single page
      (3,                       2,      False,  "multi_page"),                     # Multiple pages per worker
      (2,                       3,      False,  "more_pages"),                     # More pages than workers, within capacity
      (TEST_POOL_CAPACITY,      1,      False,  "max_capacity"),                   # Max capacity single pages
      (TEST_POOL_CAPACITY // 2, 2,      False,  "max_capacity_multi"),             # Max capacity multiple pages
      (4,                       3,      True ,  "exceeds_total"),                  # 12 pages needed, exceeds capacity
      (TEST_POOL_CAPACITY + 1,  1,      True ,  "exceeds_workers"),                # More workers than capacity
      (TEST_POOL_CAPACITY // 2, 3,      True ,  "exceeds_with_multi"),             # Exceeds capacity with multiple pages
  ],
)
# fmt: on
def test_concurrent_page_allocation_ref_counts(
    cache_ref_count,
    num_workers,
    pages_per_worker,
    expect_failure,
    case_name,
):
    allocated_pages = defaultdict(set)
    errors = []
    allocations = []

    def worker(worker_id: int):
        try:
            tokens = list(range(TEST_PAGE_SIZE * pages_per_worker))
            allocation = cache_ref_count.allocate(tokens)
            allocations.append(allocation)
            allocated_pages[worker_id] = {page.index for page in allocation.pages}
            time.sleep(random.uniform(0.001, 0.01))
        except CacheAllocationFailure as e:
            errors.append(e)
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if expect_failure:
        assert len(errors) > 0, "Expected at least one CacheAllocationFailure"
    else:
        assert not errors, f"Workers encountered errors: {errors}"
        for worker_id, pages in allocated_pages.items():
            assert (
                len(pages) == pages_per_worker
            ), f"Worker {worker_id} got {len(pages)} pages, expected {pages_per_worker}"

        all_pages = set()
        for pages in allocated_pages.values():
            assert not (
                pages & all_pages
            ), f"Found duplicate page allocation: {pages & all_pages}"
            all_pages.update(pages)

    for allocation in allocations:
        cache_ref_count.free_pages(allocation.pages)


@pytest.mark.parametrize(
    "total_pages_needed",
    [
        TEST_POOL_CAPACITY + 1,  # Just over capacity
        TEST_POOL_CAPACITY * 2,  # Double capacity
    ],
)
def test_allocation_failure_when_exhausted(cache, cache_ref_count, total_pages_needed):
    successful_allocations = []

    for _cache in (cache, cache_ref_count):
        try:
            tokens = list(range(TEST_PAGE_SIZE * total_pages_needed))
            allocation = _cache.allocate(tokens)
            successful_allocations.append(allocation)
        except CacheAllocationFailure as e:
            pass
        else:
            pytest.fail("Expected CacheAllocationFailure was not raised")
        finally:
            for alloc in successful_allocations:
                cache.free_pages(alloc.pages)


# fmt: off
@pytest.mark.parametrize(
   "tokens,expected_pages,case_name",
   [   # Tokens                                Pages  Case Name
       ([],                                    0,     "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      1,     "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           1,     "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       2,     "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       2,     "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   3,     "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       3,     "three_exact_pages"),
       (list(range(1)),                        1,     "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       1,     "almost_full_page"),
   ],
)
# fmt: on
def test_free_pages(cache, tokens, expected_pages, case_name):
    total_pages = len(cache.page_pool.attn_page_entries)

    allocation = cache.allocate(tokens)
    pages = allocation.pages

    cache.free_pages(pages)
    qsize = cache.page_pool._queue.qsize()
    assert (
        qsize == total_pages
    ), f"All pages should be freed for {case_name}, but only freed {qsize}"
