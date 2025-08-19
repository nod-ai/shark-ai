# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading

import shortfin.array as sfnp


class Allocation:
    def __init__(self, *, device, host, cache, key):
        self._device = device
        self._host = host
        self._cache = cache
        self._key = key

    @property
    def key(self):
        return self._key

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
        self._cache.release(self)

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


class DeviceArrayCache:
    def __init__(self, device, *, max_allocations=100):
        self._device = device
        self._max_allocations = max_allocations
        self._cache_lock = threading.Lock()

        self._id = 0
        self._shape_table = {}
        self._cache = {}

    def allocate(self, shape, dtype) -> Allocation:
        with self._cache_lock:
            key = self.create_key(shape=shape, dtype=dtype)

            # If we already have it allocated find the entry, possibly clear the list and return the allocation:
            if key in self._shape_table:
                idx = self._shape_table[key].pop()
                if len(self._shape_table[key]) == 0:
                    del self._shape_table[key]
                return self._cache.pop(idx)

            # If we are exceeding the recommended cache size use this as an opportunity to clean up:
            if len(self._cache) > self._max_allocations:
                # Grab the keys that should be cleaned up:
                release_count = len(self._cache) - self._max_allocations
                keys = sorted(self._cache.keys())
                to_keep = set(keys[release_count:])
                new_cache = {idx: self._cache[idx] for idx in to_keep}
                new_table = {}
                for idx in to_keep:
                    new_key = self.create_key(allocation=self._cache[idx])
                    if new_key not in new_table:
                        new_table[new_key] = []
                    new_table[new_key].append(idx)

                self._cache = new_cache
                self._shape_table = new_table

        device = sfnp.device_array.for_device(self._device, shape, dtype)
        host = device.for_transfer()

        return Allocation(device=device, host=host, cache=self, key=key)

    def create_key(self, *, allocation=None, shape=None, dtype=None):
        if allocation is not None:
            if shape is not None or dtype is not None:
                raise Exception("Cannot specify both allocation and shape/dtype")

            return allocation.key

        shape = "x".join([str(d) for d in shape])
        return f"{shape}, {dtype}"

    def release(self, allocation):
        with self._cache_lock:
            idx = self._id
            key = self.create_key(allocation=allocation)

            if key not in self._shape_table:
                self._shape_table[key] = []

            self._shape_table[key].append(idx)
            self._cache[idx] = allocation
            self._id += 1

    def free(self):
        with self._cache_lock:
            del self._cache
            self._cache = {}
