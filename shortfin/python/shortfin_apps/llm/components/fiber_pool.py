# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import shortfin as sf
from ...utils import SystemManager
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
        sysman: SystemManager,
        init_size: int,
        resizable: bool = True,
        name: str = "default-fiber-pool",
        disaggregate: bool = False,
    ):
        self.init_size: int = init_size
        self.resizable: bool = resizable
        self.sysman: LlmSystemManager = sysman
        self.name: str = name
        self.disaggregate = disaggregate
        self._fiber_pool: list[sf.Fiber] = []
        self._workers: list[sf.Worker] = []
        # Keep track of how many extra fibers were created
        # during runtime if `resizable` is set to True.
        self._extra_fibers: int = 0
        self._created_fibers_on_request: int = 0
        self._index_queue = asyncio.Queue()
        self._lock = Lock()
        self._initialize_pool()

    def _resize(self) -> tuple[int, sf.Fiber]:
        new_worker = self.sysman.ls.create_worker(
            f"{self.name}-new-worker-{self._extra_fibers}"
        )
        fiber = self.create_fiber(worker=new_worker)
        self._extra_fibers += 1
        return (
            self.size() - 1,
            fiber,
        )

    async def get(self) -> tuple[int, sf.Fiber]:
        with self._lock:
            try:
                idx = self._index_queue.get_nowait()
                self._index_queue.task_done()
                return (
                    idx,
                    self._fiber_pool[idx],
                )
            except asyncio.QueueEmpty:
                if self.resizable:
                    # Resize the fiber pool by adding a new fiber.
                    return self._resize()

                available_index = await self._index_queue.get()
                self._index_queue.task_done()
                return (available_index, self._fiber_pool[available_index])

    def _initialize_pool(self):
        with self._lock:
            for idx in range(self.init_size):
                worker = self.sysman.ls.create_worker(
                    f"{self.name}-initial-worker-{idx}"
                )
                self.create_fiber(worker=worker)
                self._index_queue.put_nowait(idx)

    def create_fiber(self, worker: sf.Worker | None = None) -> sf.Fiber:
        current_device_idx = len(self._fiber_pool) % len(self.sysman.ls.devices)
        if worker is None:
            worker = self.sysman.create_worker(
                f"{self.name}-worker-on-request-{self._created_fibers_on_request}"
            )
            self._created_fibers_on_request += 1

        devices = (
            self.sysman.ls.devices
            if not self.disaggregate
            else [self.sysman.ls.devices[current_device_idx]]
        )

        fiber = self.sysman.ls.create_fiber(worker, devices=devices)
        self._workers.append(worker)
        self._fiber_pool.append(fiber)
        return fiber

    def return_fiber(self, indices: int | list[int]):
        with self._lock:
            if not isinstance(indices, list):
                indices = [indices]
            for idx in indices:
                self._index_queue.put_nowait(idx)

    def size(self) -> int:
        """
        NOTE: Due to multiple threads accessing the fiber pool concurrently, this function
        is NOT a reliable way to find the size of the fiber pool at any given time during
        execution.
        """
        return len(self._fiber_pool)
