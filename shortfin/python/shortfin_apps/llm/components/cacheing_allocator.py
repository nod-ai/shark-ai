# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin.array as sfnp


class Allocation:
    def __init__(self, device, host, allocator):
        self._device = device
        self._host = host
        self._allocator = allocator

    @property
    def device(self):
        return self._device

    @property
    def host(self):
        return self._host

    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    @property
    def wrapped(self):
        return False

    def release(self):
        self._allocator.release(self)

    def transfer_to_device(self):
        self.host.copy_to(self.device)


class WrappedAllocation:
    def __init__(self, device):
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def host(self):
        return None

    @property
    def shape(self):
        return (
            self._device.delegate().shape
            if isinstance(self._device, sfnp.disable_barrier)
            else self._device.shape
        )

    @property
    def dtype(self):
        return (
            self._device.delegate().dtype
            if isinstance(self._device, sfnp.disable_barrier)
            else self._device.dtype
        )

    @property
    def wrapped(self):
        return True

    def transfer_to_device(self):
        assert False

    def release(self):
        pass


def _shape_matches(a, b):
    if len(a) != len(b):
        return False

    return all([_a == _b for _a, _b in zip(a, b)])


class CacheingAllocator:
    def __init__(self, device, *, max_allocations=100):
        self._device = device
        self._cache = []
        self._max_allocations = max_allocations

    def allocate(self, shape, dtype):
        for i, allocation in enumerate(self._cache):
            if _shape_matches(allocation.shape, shape) and (allocation.dtype == dtype):
                return self._cache.pop(i)

        if len(self._cache) > self._max_allocations:
            diff = len(self._cache) - self._max_allocations
            to_del = self._cache[:diff]
            self._cache = self._cache[diff:]
            del to_del

        device = sfnp.device_array.for_device(self._device, shape, dtype)
        host = device.for_transfer()

        return Allocation(device, host, self)

    def release(self, allocation):
        self._cache.append(allocation)

    def free(self):
        del self._cache
        self._cache = []
