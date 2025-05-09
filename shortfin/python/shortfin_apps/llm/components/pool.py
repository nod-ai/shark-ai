# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio


class Pool:
    def __init__(self, worker_count, service):
        self.queue = asyncio.Queue()
        self.worker_count = worker_count
        self.service = service

        self.fibers = []
        for i in range(worker_count):
            worker = self.service.sysman.ls.create_worker(f"pool-main-{i}")
            fiber = self.service.sysman.ls.create_fiber(worker)
            self.fibers.append(fiber)

    async def wait(self):
        await self.queue.join()

    def enqueue(self, task):
        self.queue.put_nowait(task)

    def start(self):
        async def worker(queue, fiber):
            while True:
                task = await queue.get()
                task.fiber = fiber
                await task.do_work()
                queue.task_done()

        self.workers = [
            asyncio.create_task(worker(self.queue, self.fibers[i])) for i in range(self.worker_count)
        ]

    def shutdown(self):
        [worker.cancel() for worker in self.workers]


class PoolTask:
    async def do_work(self):
        ...
