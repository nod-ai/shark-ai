# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio


class Pool:
    def __init__(self, worker_count):
        self.queue = asyncio.Queue()
        self.worker_count = worker_count

    async def wait(self):
        await self.queue.join()

    def enqueue(self, task):
        self.queue.put_nowait(task)

    def start(self):
        async def worker(queue):
            while True:
                task = await queue.get()
                await task.do_work()
                queue.task_done()

        self.workers = [
            asyncio.create_task(worker(self.queue)) for _ in range(self.worker_count)
        ]

    def shutdown(self):
        [worker.cancel() for worker in self.workers]


class PoolTask:
    async def do_work(self):
        ...
