# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from shortfin_apps.llm.components.fiber_pool import FiberPool
from shortfin_apps.llm.components.manager import LlmSystemManager

from concurrent.futures import ThreadPoolExecutor

import shortfin as sf
import asyncio
import time

FIBER_POOL_INIT_SIZE: int = 16
DELAY_TOLERANCE = 2.0000


class MockSfProcess(sf.Process):
    def __init__(self, fiber_pool: FiberPool, fiber: tuple[int, sf.Fiber]):
        super().__init__(fiber=fiber[1])
        self.pool = fiber_pool
        self.fiber_idx = fiber[0]

    async def run(self):
        # Simulate long-running work.
        await asyncio.sleep(2)
        self.pool.return_fiber(self.fiber_idx)

    @staticmethod
    async def toplevel(processes):
        for proc in processes:
            proc.launch()

        await asyncio.gather(*processes)


class BlockingMockSfProcess(sf.Process):
    def __init__(self, fiber_pool: FiberPool, fiber: tuple[int, sf.Fiber]):
        super().__init__(fiber=fiber[1])
        self.pool = fiber_pool
        self.fiber_idx = fiber[0]

    async def run(self):
        # Block the whole event loop.
        time.sleep(1)
        self.pool.return_fiber(self.fiber_idx)

    @staticmethod
    async def toplevel(processes, pool: FiberPool):
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
        sysman=sysman,
        init_size=FIBER_POOL_INIT_SIZE,
        resizable=True,
    )
    return resizable_fiber_pool


@pytest.fixture
def static_fiber_pool(sysman) -> FiberPool:
    static_fiber_pool = FiberPool(
        sysman=sysman,
        init_size=FIBER_POOL_INIT_SIZE,
        resizable=False,
    )

    return static_fiber_pool


def test_fiber_pool_init_size(fiber_pool: FiberPool, sysman: LlmSystemManager):
    """
    Test the initialization size of the FiberPool.
    """
    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE


@pytest.mark.asyncio
async def test_fiber_pool_resize(fiber_pool: FiberPool, sysman: LlmSystemManager):
    """
    Test that the FiberPool resizes correctly when there is a shortage
    of available fibers.
    """
    extra_fibers = 2
    fibers = []
    for _ in range(FIBER_POOL_INIT_SIZE + extra_fibers):
        idx, fiber = await fiber_pool.get()
        fibers.append(
            (
                idx,
                fiber,
            )
        )

    procs = [
        MockSfProcess(fiber_pool, fibers[i])
        for i in range(FIBER_POOL_INIT_SIZE + extra_fibers)
    ]

    sysman.ls.run(MockSfProcess.toplevel(procs))
    assert fiber_pool.size() == FIBER_POOL_INIT_SIZE + extra_fibers


@pytest.mark.asyncio
async def test_fiber_pool_parallelism(fiber_pool: FiberPool, sysman: LlmSystemManager):
    fibers = []
    for _ in range(FIBER_POOL_INIT_SIZE):
        idx, fiber = await fiber_pool.get()
        fibers.append(
            (
                idx,
                fiber,
            )
        )
    procs = [
        BlockingMockSfProcess(fiber_pool, fibers[i])
        for i in range(FIBER_POOL_INIT_SIZE)
    ]

    # If there were true parallelism in the fiber pool, and the fibers are not holding
    # the GIL, we should see them running blocking tasks in parallel. time.sleep() should
    # block the whole event loop of the Process, which means that unless there was true parallelism
    # we would see the time of execution come out to be somewhere near FIBER_POOL_INIT_SIZE when
    # each process waits for 1.0 second. We assert that it is less than 2 seconds, allowing 1 second
    # as tolerance.
    start = time.time()
    sysman.ls.run(BlockingMockSfProcess.toplevel(procs, fiber_pool))
    end = time.time()
    diff = end - start
    assert diff < DELAY_TOLERANCE
