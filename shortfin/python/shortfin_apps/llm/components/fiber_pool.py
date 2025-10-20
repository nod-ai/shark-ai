# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import asyncio
import logging

import shortfin as sf

from threading import Lock

from .manager import LlmSystemManager


logger = logging.getLogger(__name__)


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

        # Name mangle to make outside access harder.
        self.__fiber_pool: list[sf.Fiber] = []
        self.__workers: list[sf.Worker] = []
        # Keep track of how many extra fibers were created
        # during runtime if `resizable` is set to True.
        self.__extra_fibers: int = 0
        self.__index_queue = asyncio.Queue()
        # Any code that modifies the index_queue or the fiber_pool
        # needs to be locked. asyncio.Queue is not thread-safe, so
        # this is required to avoid issues like new fibers with the
        # same name as existing ones.
        self.__lock = Lock()
        self.__initialize_pool()

    async def get(self) -> tuple[int, sf.Fiber]:
        try:
            with self.__lock:
                idx = self.__index_queue.get_nowait()
                return (
                    idx,
                    self.__fiber_pool[idx],
                )
        except asyncio.QueueEmpty:
            if self.resizable:
                with self.__lock:
                    # Resize the fiber pool by adding a new fiber.
                    new_worker = self.sysman.ls.create_worker(
                        f"{self.name}-new-worker-{self.__extra_fibers}"
                    )
                    self.__workers.append(new_worker)

                    fiber = self.sysman.ls.create_fiber(new_worker)
                    self.__fiber_pool.append(fiber)
                    self.__extra_fibers += 1
                    return [self.size() - 1, fiber]

            logger.warning(
                f"Fiber pool {self.name} exhausted. Consider increasing its size or making it resizable."
            )
            available_index = await self.__index_queue.get()
            return (available_index, self.__fiber_pool[available_index])

    def pool(self) -> list[sf.Fiber]:
        return self.__fiber_pool

    def __initialize_pool(self):
        with self.__lock:
            for idx in range(self.init_size):
                worker = self.sysman.ls.create_worker(f"{self.name}-init-worker-{idx}")
                self.__workers.append(worker)

                fiber = self.sysman.ls.create_fiber(worker)
                self.__fiber_pool.append(fiber)
                assert idx < self.size()
                self.__index_queue.put_nowait(idx)

    def return_fiber(self, indices: int | list[int]):
        if not isinstance(indices, list):
            indices = [indices]
        with self.__lock:
            for idx in indices:
                self.__index_queue.put_nowait(idx)

    def size(self) -> int:
        return len(self.__fiber_pool)
