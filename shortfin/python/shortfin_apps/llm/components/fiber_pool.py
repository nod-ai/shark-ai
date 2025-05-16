# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import shortfin as sf
from .manager import LlmSystemManager
import asyncio
from threading import Lock


class FiberPool:
    """
    Implements a pool of fibers that can be accessed on-demand.
    The primary reason behind this implementation is to be prevent the main thread
    from keeping busy with CPU work and starving the GPU of tasks to do.

    NOTE: This class will eventually need support for mapping fibers to distinct logical
    devices once multiple HIP stream support is implemented.
    """

    def __init__(
        self,
        sysman: LlmSystemManager,
        init_size: int,
        resizable: bool = True,
        name: str = "default-fiber-pool",
    ):
        self.init_size: int = init_size
        self.resizable: bool = resizable
        self.sysman: LlmSystemManager = sysman
        self.name: str = name

        self._fiber_pool: list[sf.Fiber] = []
        self._workers: list[sf.Worker] = []
        # Keep track of how many extra fibers were created
        # during runtime if `resizable` is set to True.
        self._extra_fibers: int = 0
        self._index_queue = asyncio.Queue()
        self._lock = Lock()
        self._initialize_pool()

    def resize(self):
        new_worker = self.sysman.ls.create_worker(
            f"{self.name}-new-worker-{self._extra_fibers}"
        )
        self._workers.append(new_worker)
        fiber = self.sysman.ls.create_fiber(new_worker)
        self._fiber_pool.append(fiber)
        self._extra_fibers += 1

        return [self.size() - 1, fiber]

    async def get(self) -> tuple[int, sf.Fiber]:
        with self._lock:
            try:
                idx = self._index_queue.get_nowait()
                return (
                    idx,
                    self._fiber_pool[idx],
                )
            except asyncio.QueueEmpty:
                if self.resizable:
                    # Resize the fiber pool by adding a new fiber.
                    return self.resize()

                available_index = await self._index_queue.get()
                return (available_index, self._fiber_pool[available_index])

    def pool(self) -> list[sf.Fiber]:
        return self._fiber_pool

    def _initialize_pool(self):
        with self._lock:
            for idx in range(self.init_size):
                worker = self.sysman.ls.create_worker(f"{self.name}-init-worker-{idx}")
                self._workers.append(worker)
                fiber = self.sysman.ls.create_fiber(worker)
                self._fiber_pool.append(fiber)
                assert idx < self.size()
                self._index_queue.put_nowait(idx)

    def return_fiber(self, indices: int | list[int]):
        with self._lock:
            if not isinstance(indices, list):
                indices = [indices]
            for idx in indices:
                self._index_queue.put_nowait(idx)

    def size(self) -> int:
        return len(self._fiber_pool)


class DisaggregatedFiberPool(FiberPool):
    def __init__(
        self,
        sysman: LlmSystemManager,
        init_size: int,
        resizable: bool = True,
        name: str = "default-disagg-fiber-pool",
    ):
        super().__init__(
            sysman=sysman,
            init_size=init_size,
            resizable=resizable,
            name=name,
        )

    def resize(self):
        devices = self.sysman.ls.devices
        num_devices = len(devices)
        new_worker = self.sysman.ls.create_worker(
            f"{self.name}-new-worker-{self._extra_fibers}"
        )
        self._workers.append(new_worker)

        fiber = self.sysman.ls.create_fiber(
            new_worker, devices=[devices[self.size() % num_devices]]
        )
        self._fiber_pool.append(fiber)
        self._extra_fibers += 1
        return [self.size() - 1, fiber]

    def _initialize_pool(self):
        with self._lock:
            devices = self.sysman.ls.devices
            num_devices = len(devices)
            for idx in range(self.init_size):
                worker = self.sysman.ls.create_worker(f"{self.name}-init-worker-{idx}")
                self._workers.append(worker)

                fiber = self.sysman.ls.create_fiber(
                    worker, devices=[devices[idx % num_devices]]
                )
                self._fiber_pool.append(fiber)
                assert idx < self.size()
                self._index_queue.put_nowait(idx)
