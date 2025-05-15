# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import shortfin as sf
from .manager import LlmSystemManager
import asyncio


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

        self.__initialize_pool()

    async def get(self) -> sf.Fiber | None:
        if len(self.__fiber_pool) > 0:
            return self.__fiber_pool.pop()

        if self.resizable:
            # Resize the fiber pool by adding a new fiber.
            new_worker = self.sysman.ls.create_worker(
                f"{self.name}-new-worker-{self.__extra_fibers}"
            )
            self.__workers.append(new_worker)

            fiber = self.sysman.ls.create_fiber(new_worker)
            self.__extra_fibers += 1
            return fiber

        # TODO(vinayakdsci): This is a rather inelegant solution to waiting for a fiber
        # to become available, because of the spin loop. Figure out a cleaner way to block
        # until fibers are available, and recommend using a resizable FiberPool.
        while self.size() < 1:
            await asyncio.sleep(0.001)

        return self.__fiber_pool.pop()

    def pool(self) -> list[sf.Fiber]:
        return self.__fiber_pool

    def __initialize_pool(self):
        for idx in range(self.init_size):
            worker = self.sysman.ls.create_worker(f"{self.name}-init-worker-{idx}")
            self.__workers.append(worker)

            fiber = self.sysman.ls.create_fiber(worker)
            self.__fiber_pool.append(fiber)

    def return_fiber(self, fibers: sf.Fiber | list[sf.Fiber]):
        if not isinstance(fibers, list):
            fibers = [fibers]
        for fiber in fibers:
            self.__fiber_pool.append(fiber)

    def size(self) -> int:
        return len(self.__fiber_pool)
