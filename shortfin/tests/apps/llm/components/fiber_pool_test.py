# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from shortfin_apps.llm.components.fiber_pool import FiberPool
from shortfin_apps.llm.components.manager import LlmSystemManager

import shortfin as sf
import asyncio

FIBER_POOL_INIT_SIZE: int = 16


class MockSfProcess(sf.Process):
    def __init__(self, fiber_pool: FiberPool, fiber: sf.Fiber):
        super().__init__(fiber=fiber)
        self.pool = fiber_pool

    async def run(self):
        await asyncio.sleep(0.1)
        self.pool.return_fiber(self.fiber)

    @staticmethod
    async def toplevel(processes):
        for proc in processes:
            proc.launch()

        await asyncio.gather(*processes)


@pytest.fixture
def sysman() -> LlmSystemManager:
    sysman = LlmSystemManager(device="local-task")
    return sysman


@pytest.fixture
def fiber_pool(sysman) -> FiberPool:
    resizable_fiber_pool = FiberPool(
        sysman=sysman, init_size=FIBER_POOL_INIT_SIZE, resizable=True
    )
    return resizable_fiber_pool


def test_fiber_pool_init_size(fiber_pool: FiberPool, sysman: LlmSystemManager):
    """
    Test the initialization size of the FiberPool.
    """
    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE


def test_fiber_pool_multiple_process(fiber_pool: FiberPool, sysman: LlmSystemManager):
    """
    Test the usage of the FiberPool when it is distributed among multiple processes.
    """
    procs = [MockSfProcess(fiber_pool, fiber_pool.get()) for _ in range(5)]

    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE - len(procs)
    sysman.ls.run(MockSfProcess.toplevel(procs))
    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE


def test_fiber_pool_resize(fiber_pool: FiberPool, sysman: LlmSystemManager):
    """
    Test that the FiberPool resizes correctly when there is a shortage
    of available fibers.
    """
    extra_fibers = 2
    procs = [
        MockSfProcess(fiber_pool, fiber_pool.get())
        for _ in range(FIBER_POOL_INIT_SIZE + extra_fibers)
    ]

    assert fiber_pool.size() == 0
    sysman.ls.run(MockSfProcess.toplevel(procs))
    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE + extra_fibers
