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
    allocation = cache.acquire_pages_for_tokens(tokens)
    pages = allocation.pages
    assert len(pages) == expected_pages, f"Failed for case: {case_name}"
    allocation.release_pages()


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
    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    ref_counts = cache_ref_count.ref_counts
    assert (
        len(ref_counts) == TEST_POOL_CAPACITY
    ), f"ref_counts failure for case: {case_name}"

    pages = allocation.pages
    assert len(pages) == expected_pages, f"Allocation failed for case: {case_name}"
    for page in pages:
        assert ref_counts[page.index] == 1


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
            allocation = cache.acquire_pages_for_tokens(tokens)
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
        allocation.release_pages()


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
            allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
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
            for page in pages:
                assert cache_ref_count.ref_counts[page] == 1

    for allocation in allocations:
        allocation.release_pages()

    for pages in allocated_pages.values():
        for page in pages:
            assert cache_ref_count.ref_counts[page] == 0


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
            allocation = _cache.acquire_pages_for_tokens(tokens)
            successful_allocations.append(allocation)
        except CacheAllocationFailure as e:
            pass
        else:
            pytest.fail("Expected CacheAllocationFailure was not raised")
        finally:
            for alloc in successful_allocations:
                alloc.release_pages()


# fmt: off
@pytest.mark.parametrize(
   "tokens,case_name",
   [   # Tokens                                Case Name
       ([],                                    "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       "three_exact_pages"),
       (list(range(1)),                        "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       "almost_full_page"),
   ],
)
# fmt: on
def test_increment_pages(cache_ref_count, tokens, case_name):
    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    ref_counts = cache_ref_count.ref_counts

    pages = allocation.pages
    cache_ref_count.increment_pages(pages)
    for page in pages:
        assert (
            ref_counts[page.index] == 2
        ), f"Error incrementing pages for {case_name}. Got {ref_counts[page.index]}, expected 2."


@pytest.mark.parametrize("num_workers", [2, 5, 10, 20])
def test_concurrent_increment_pages(cache_ref_count, num_workers):
    pages = cache_ref_count.page_pool.attn_page_entries
    ref_counts = cache_ref_count.ref_counts

    def increment():
        cache_ref_count.increment_pages(pages)

    threads = [threading.Thread(target=increment) for _ in range(num_workers)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Validate that each page's ref_count is correctly incremented
    for page in pages:
        expected_count = num_workers
        assert (
            ref_counts[page.index] == expected_count
        ), f"Thread safety issue w/ num_workers == {num_workers}: expected {expected_count}, got {ref_counts[page.index]} for page {page.index}."


# fmt: off
@pytest.mark.parametrize(
   "tokens,case_name",
   [   # Tokens                                Case Name
       ([],                                    "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       "three_exact_pages"),
       (list(range(1)),                        "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       "almost_full_page"),
   ],
)
# fmt: on
def test_decrement_pages(cache_ref_count, tokens, case_name):
    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    ref_counts = cache_ref_count.ref_counts

    pages = allocation.pages
    cache_ref_count.decrement_pages(pages)
    for page in pages:
        assert (
            ref_counts[page.index] == 0
        ), f"Error incrementing pages for {case_name}. Got {ref_counts[page.index]}, expected 2."


# fmt: off
@pytest.mark.parametrize(
   "tokens,case_name",
   [   # Tokens                                Case Name
       ([],                                    "empty_token_list"),
       (list(range(TEST_PAGE_SIZE // 2)),      "partial_page"),
       (list(range(TEST_PAGE_SIZE)),           "exact_page"),
       (list(range(TEST_PAGE_SIZE + 1)),       "just_over_one_page"),
       (list(range(TEST_PAGE_SIZE * 2)),       "multiple_exact_pages"),
       (list(range(TEST_PAGE_SIZE * 2 + 1)),   "multiple_pages_with_remainder"),
       (list(range(TEST_PAGE_SIZE * 3)),       "three_exact_pages"),
       (list(range(1)),                        "single_token"),
       (list(range(TEST_PAGE_SIZE - 1)),       "almost_full_page"),
   ],
)
# fmt: on
def test_decrement_pages_return_empty_pages(cache_ref_count, tokens, case_name):
    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    ref_counts = cache_ref_count.ref_counts

    pages = allocation.pages
    empty_pages = cache_ref_count.decrement_pages(pages, return_empty_pages=True)
    assert len(empty_pages) == len(pages)
    for page in pages:
        assert (
            ref_counts[page.index] == 0
        ), f"Error incrementing pages for {case_name}. Got {ref_counts[page.index]}, expected 2."


@pytest.mark.parametrize("num_workers", [2, 5, 10, 20])
def test_concurrent_decrement_pages(cache_ref_count, num_workers):
    pages = cache_ref_count.page_pool.attn_page_entries
    ref_counts = cache_ref_count.ref_counts

    for page in pages:
        ref_counts[page.index] = num_workers

    def decrement():
        cache_ref_count.decrement_pages(pages)

    threads = [threading.Thread(target=decrement) for _ in range(num_workers)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Validate that each page's ref_count is correctly incremented
    for page in pages:
        expected_count = 0
        assert (
            ref_counts[page.index] == expected_count
        ), f"Thread safety issue w/ num_workers == {num_workers}: expected {expected_count}, got {ref_counts[page.index]} for page {page.index}."


@pytest.mark.parametrize("num_workers", [2, 5, 10, 20])
@pytest.mark.parametrize("order", ["increment_first", "decrement_first", "mixed"])
def test_concurrent_increment_decrement_pages(cache_ref_count, num_workers, order):
    pages = cache_ref_count.page_pool.attn_page_entries
    ref_counts = cache_ref_count.ref_counts

    def increment():
        cache_ref_count.increment_pages(pages)

    def decrement():
        cache_ref_count.decrement_pages(pages)

    threads = []

    if order == "increment_first":
        # Start all increment threads first, then decrement threads
        threads.extend(threading.Thread(target=increment) for _ in range(num_workers))
        threads.extend(threading.Thread(target=decrement) for _ in range(num_workers))
    elif order == "decrement_first":
        # Start all decrement threads first, then increment threads
        threads.extend(threading.Thread(target=decrement) for _ in range(num_workers))
        threads.extend(threading.Thread(target=increment) for _ in range(num_workers))
    elif order == "mixed":
        # Interleave increments and decrements randomly
        operations = [increment] * num_workers + [decrement] * num_workers
        random.shuffle(operations)
        threads.extend(threading.Thread(target=op) for op in operations)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    for page in pages:
        expected_count = 0
        assert (
            ref_counts[page.index] == expected_count
        ), f"Thread safety issue in {order} order w/ num_workers == {num_workers}: expected {expected_count}, got {ref_counts[page.index]} for page {page.index}."


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

    allocation = cache.acquire_pages_for_tokens(tokens)
    pages = allocation.pages

    cache.free_pages(pages)
    qsize = cache.page_pool._queue.qsize()
    assert (
        qsize == total_pages
    ), f"All pages should be freed for {case_name}, but only freed {qsize}"


@pytest.mark.parametrize(
    "scenario", ["no_pages_freed", "some_pages_freed", "all_pages_freed"]
)
# fmt: off
@pytest.mark.parametrize(
    "tokens,expected_pages,case_name",
    [  # Tokens                                Pages  Case Name
        ([], 0, "empty_token_list"),
        (list(range(TEST_PAGE_SIZE // 2)), 1, "partial_page"),
        (list(range(TEST_PAGE_SIZE)), 1, "exact_page"),
        (list(range(TEST_PAGE_SIZE + 1)), 2, "just_over_one_page"),
        (list(range(TEST_PAGE_SIZE * 2)), 2, "multiple_exact_pages"),
        (list(range(TEST_PAGE_SIZE * 2 + 1)), 3, "multiple_pages_with_remainder"),
        (list(range(TEST_PAGE_SIZE * 3)), 3, "three_exact_pages"),
        (list(range(1)), 1, "single_token"),
        (list(range(TEST_PAGE_SIZE - 1)), 1, "almost_full_page"),
    ],
)
# fmt: on
def test_free_pages_use_ref_count(
    cache_ref_count, scenario, tokens, expected_pages, case_name
):
    total_pages = len(cache_ref_count.page_pool.attn_page_entries)
    ref_counts = cache_ref_count.ref_counts

    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    pages = allocation.pages

    if scenario == "no_pages_freed":
        # Artificially increment ref_count so pages should not be freed
        cache_ref_count.increment_pages(pages)

    elif scenario == "some_pages_freed":
        pages_to_increment = [page for i, page in enumerate(pages) if i % 2 == 0]
        # Increment ref_count for half of the pages (so only half should be freed)
        cache_ref_count.increment_pages(pages_to_increment)

    # Attempt to free pages
    cache_ref_count.free_pages(pages)

    if scenario == "no_pages_freed":
        # Ensure no pages were freed
        for page in pages:
            assert (
                ref_counts[page.index] > 0
            ), f"Page {page.index} should not have been freed in scenario {scenario}, case_name {case_name}"
        assert cache_ref_count.page_pool._queue.qsize() == total_pages - len(pages)

    elif scenario == "some_pages_freed":
        # Ensure only some pages were freed
        freed_pages = 0
        for page in pages:
            if ref_counts[page.index] == 0:
                freed_pages += 1
        assert (
            freed_pages == len(pages) // 2
        ), f"Expected some pages to be freed in {scenario}, case_name {case_name}"
        assert cache_ref_count.page_pool._queue.qsize() == total_pages - (
            len(pages) - freed_pages
        )

    else:  # "all_pages_freed"
        # Ensure all pages were freed
        for page in pages:
            assert (
                ref_counts[page.index] == 0
            ), f"Page {page.index} was not freed in scenario {scenario}, case_name {case_name}"
        assert cache_ref_count.page_pool._queue.qsize() == total_pages


# fmt: off
@pytest.mark.parametrize(
    "tokens,expected_pages,case_name",
    [  # Tokens                                Pages  Case Name
        (list(range(TEST_PAGE_SIZE // 2)), 1, "partial_page"),
        (list(range(TEST_PAGE_SIZE)), 1, "exact_page"),
        (list(range(TEST_PAGE_SIZE + 1)), 2, "just_over_one_page"),
        (list(range(TEST_PAGE_SIZE * 2)), 2, "multiple_exact_pages"),
        (list(range(TEST_PAGE_SIZE * 2 + 1)), 3, "multiple_pages_with_remainder"),
        (list(range(TEST_PAGE_SIZE * 3)), 3, "three_exact_pages"),
        (list(range(1)), 1, "single_token"),
        (list(range(TEST_PAGE_SIZE - 1)), 1, "almost_full_page"),
    ],
)
# fmt: on
@pytest.mark.asyncio
@pytest.mark.xfail(reason="xfailed for flakiness on the new-pages-should-be-all-1s assertion. See https://github.com/nod-ai/shark-ai/issues/1176")
async def test_fork_pages(cache_ref_count, tokens, expected_pages, case_name):
    ref_counts = cache_ref_count.ref_counts

    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    pages = allocation.pages

    last_page = pages[-1]
    # Update index of last page in `page_tables` with all `1s`
    page_tables = cache_ref_count.page_pool.page_tables
    with page_tables[0].view(last_page.index).map(discard=True) as m:
        m.fill(1)

    new_allocation = cache_ref_count.fork_pages(pages)
    new_pages = new_allocation.pages
    # The await here allows the data to finish copying over,
    # before we check the values.
    await asyncio.sleep(0.1)
    pages.pop(-1)
    new_last_page = new_pages.pop(-1)

    assert (
        last_page.index != new_last_page.index
    ), f"Fork Error in {case_name}: Last pages should be different."

    # All pages are shared, except for the last one
    for i in range(len(pages)):
        assert (
            pages[i] == new_pages[i]
        ), f"Fork Error in {case_name}: Page {i} should be shared."

    # Ref counts should be `2` for shared pages,
    # and `1` for last pages
    for page in pages:
        assert (
            ref_counts[page.index] == 2
        ), f"Fork Error in {case_name}: Page {page.index} should have ref_count 2."

    assert (
        ref_counts[last_page.index] == 1
    ), f"Fork Error in {case_name}: Last page should have ref_count 1."
    assert (
        ref_counts[new_last_page.index] == 1
    ), f"Fork Error in {case_name}: New last page should have ref_count 1."

    original_page_table = page_tables[0].view(last_page.index).items.tolist()
    new_page_table = page_tables[0].view(new_last_page.index).items.tolist()

    assert all(
        val == 1.0 for val in new_page_table
    ), f"Fork Error in {case_name}: New last page should be filled with 1s."
    assert (
        original_page_table == new_page_table
    ), f"Fork Error in {case_name}: Page data should be the same."


@pytest.mark.asyncio
async def test_fork_pages_allocation_error(cache_ref_count):
    # Use all pages
    tokens = list(range(TEST_PAGE_SIZE * TEST_POOL_CAPACITY))

    allocation = cache_ref_count.acquire_pages_for_tokens(tokens)
    pages = allocation.pages
    assert all(
        val == 1 for val in cache_ref_count.ref_counts
    ), "All pages should be in use."

    # Should throw an allocation error when forking
    with pytest.raises(CacheAllocationFailure):
        cache_ref_count.fork_pages(pages)
