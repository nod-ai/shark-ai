# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.llm.components.cacheing_allocator import CacheingAllocator

import shortfin.array as sfnp


def test_allocate(generic_device):
    allocator = CacheingAllocator(generic_device)
    allocation0 = allocator.allocate((1, 2, 3), sfnp.int64)

    assert allocation0.shape[0] == 1
    assert allocation0.shape[1] == 2
    assert allocation0.shape[2] == 3
    assert allocation0.dtype == sfnp.int64


def test_release_allocate(generic_device):
    allocator = CacheingAllocator(generic_device)
    allocation0 = allocator.allocate((1, 2, 3), sfnp.int64)

    allocator.release(allocation0)
    allocation1 = allocator.allocate((1, 2, 3), sfnp.int64)

    assert allocation0.device == allocation1.device
    assert allocation0.host == allocation1.host

    assert allocation1.shape[0] == 1
    assert allocation1.shape[1] == 2
    assert allocation1.shape[2] == 3
    assert allocation1.dtype == sfnp.int64


def test_release_allocate_diff(generic_device):
    allocator = CacheingAllocator(generic_device)
    allocation0 = allocator.allocate((1, 2, 3), sfnp.int64)

    allocator.release(allocation0)
    allocation1 = allocator.allocate((1, 2, 4), sfnp.int64)

    assert allocation0.device != allocation1.device
    assert allocation0.host != allocation1.host


def test_release_allocate_multiple(generic_device):
    allocator = CacheingAllocator(generic_device)
    allocation0 = []
    allocation1 = []

    for i in range(10):
        allocation0.append(allocator.allocate((1, 2, 3), sfnp.int64))

    for allocation in allocation0[::-1]:
        allocator.release(allocation)

    for i in range(10):
        allocation1.append(allocator.allocate((1, 2, 3), sfnp.int64))

    allocation1.reverse()

    for a, b in zip(allocation0, allocation1):
        assert a.device == b.device
        assert a.host == b.host

    for i in range(10):
        for j in range(10):
            if i == j:
                continue

            assert allocation0[i].device != allocation1[j].device
            assert allocation0[i].host != allocation1[j].host


def test_release_allocate_limit(generic_device):
    allocator = CacheingAllocator(generic_device, max_allocations=1)

    allocation0 = allocator.allocate((1, 2, 3), sfnp.int64)
    allocation1 = allocator.allocate((1, 2, 3), sfnp.int64)

    allocator.release(allocation0)
    allocator.release(allocation1)

    # An uncached allocation is used to flush the cache.
    flush = allocator.allocate((1, 2, 4), sfnp.int64)

    allocation2 = allocator.allocate((1, 2, 3), sfnp.int64)
    allocation3 = allocator.allocate((1, 2, 3), sfnp.int64)

    assert allocation0.device != allocation3.device
    assert allocation0.host != allocation3.host

    assert allocation1.device == allocation2.device
    assert allocation1.host == allocation2.host
