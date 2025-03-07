from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import threading
import logging
import shortfin as sf
import shortfin.array as sfnp
from dataclasses import dataclass

from ..config_struct import human_size
import math

import time

logger = logging.getLogger(__name__)


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
class PageInfo:
    """
    Page index with some metadata about its contents.
    """

    index: int
    pool: PagePool


@dataclass
class PagePoolConfig:
    """
    Hyperparameters for the page pool.
    """

    dtype: sf.dtype
    alloc_page_count: int

    paged_kv_block_size_elements: int  # size of a single page as # of elements
    # (e.g. one configuration for llama3.1 8b hax 32x2x16x8x128=1048576 elements where:
    # 32: number of transformer blocks
    # 2: one for k + one for v
    # 16: tokens per page
    # 8: head count (32 heads, but every 4 heads share the same kv buffer)
    # 128: hidden dimension


class PagePool:
    """Page table based attention cache.

    While internal to a model, the cache is organized with additional structure
    per page, outside of the model, it is just a list of pages of a certain
    element type and number of elements (all inner dims are flattened).

    One page table is allocated per device in a fiber. Currently, this is a
    dense allocation with committed memory but in the future, we may just
    allocate the address space and lazily populate it with committed memory.

    The cache is unique because usage of it can span fibers and concurrency
    is implicitly managed at the block level (i.e. freshly acquired blocks
    are assumed to be uninitialized and available immediately for use).

    It is initialized with a discrete list of fiberd devices from a fiber but
    cache usage can be done from any fiber which includes those devices.

    In addition to supporting paged attention standalone, this also serves
    as the array / buffer allocation layer for radix attention described in
    `radix_tree.py`.
    """

    def __init__(self, *, devices: Sequence[sf.ScopedDevice], config: PagePoolConfig):
        self._lock = threading.Lock()
        self._devices = list(devices)
        self._config = config
        self._page_tables: list[sf.array.device_array] = []
        self._ref_counts: dict[int, RefCount] = {}

        # Setup accounting structs.
        self.attn_page_entries = [
            PageInfo(index=i, pool=self) for i in range(self._config.alloc_page_count)
        ]

        self._available_pages = list(self.attn_page_entries)

        # Initialize a page table on each device.
        page_table_shape = [
            self._config.alloc_page_count,
            self._config.paged_kv_block_size_elements // len(devices),
        ]
        for device in devices:
            logging.info(
                "Allocating page table (shape=%r, dtype=%r, size=%s) on %r",
                page_table_shape,
                self._config.dtype,
                human_size(config.dtype.compute_dense_nd_size(page_table_shape)),
                device,
            )
            page_table = sf.array.device_array.for_device(
                device, page_table_shape, self._config.dtype
            )
            page_table_host = page_table.for_transfer()
            with page_table_host.map(discard=True) as m:
                m.fill(0)
            page_table_host.copy_to(page_table)
            self._page_tables.append(page_table)

    def acquire_free_pages(self, count: int) -> list[PageInfo] | None:
        with self._lock:
            available = len(self._available_pages)
            if count > available:
                return None
            pages = []
            for _ in range(count):
                available_page = self._available_pages.pop()
                if available_page.index not in self._ref_counts:
                    self._ref_counts[available_page.index] = RefCount()
                self._ref_counts[available_page.index].increment()
                pages.append(available_page)
            return pages

    def retain_count(self, page) -> int:
        with self._lock:
            if page.index in self._ref_counts:
                return self._ref_counts[page.index].count
            return 0

    def retain_pages(self, pages: list[PageInfo]):
        with self._lock:
            for page in pages:
                self._ref_counts[page.index].increment()

    def free_pages(self, pages: list[PageInfo]):
        with self._lock:
            freed_pages = []
            for page in pages:
                self._ref_counts[page.index].decrement()
                if self._ref_counts[page.index].is_empty():
                    freed_pages.append(page)
            self._available_pages.extend(freed_pages)
            logger.info(f"After freeing Cache Pages: {str(self)}")

    def is_available(self, page: PageInfo) -> bool:
        return page in self._available_pages

    def copy_page(self, src_page: PageInfo) -> PageInfo:
        """
        Copy a page's contents to a new page.

        Args:
            src_page: Source page to copy from

        Returns:
            New PageInfo containing the copied data
        """
        # Allocate new page
        (dst_page,) = self.acquire_free_pages(1)

        # fill src page with data

        # Copy the data on each device
        for page_table in self._page_tables:
            # View of source and destination pages
            src_view = page_table.view(src_page.index)
            dst_view = page_table.view(dst_page.index)
            # Copy the data
            dst_view.copy_from(src_view)

        return dst_page

    def __repr__(self):
        # No need to lock for repr (list is internally synchronized).
        free_pages = len(self._available_pages)
        total_pages = len(self.attn_page_entries)
        return (
            f"PagePool({total_pages - free_pages}/{total_pages} pages in use: "
            f"{100.0 * free_pages / total_pages}% free)"
        )


############################## begin radix attention
