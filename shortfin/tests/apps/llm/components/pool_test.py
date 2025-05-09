from shortfin_apps.llm.components.pool import Pool, PoolTask

import asyncio


class Task(PoolTask):
    def __init__(self, *, name="", dependent=None, out_queue=None):
        self.name = name
        self.event = asyncio.Event()
        self.dependent = None
        self.out_queue = out_queue

    def set_dependent(self, dependent):
        self.dependent = dependent

    async def do_work(self):
        if self.dependent is not None:
            await self.dependent.wait()
        await self.out_queue.put(self.name)
        self.event.set()

    def is_done(self):
        return self.event.is_set()


def test_pool_event():
    async def async_test():
        out_queue = asyncio.Queue()

        task0 = Task(name="task-0", out_queue=out_queue)
        task1 = Task(name="task-1", out_queue=out_queue)
        task2 = Task(name="task-2", out_queue=out_queue)

        task0.set_dependent(task1.event)
        task1.set_dependent(task2.event)

        pool = Pool(worker_count=4)
        pool.start()
        pool.enqueue(task0)
        pool.enqueue(task1)
        await asyncio.sleep(0.01)

        assert not task0.is_done()
        assert not task1.is_done()
        assert not task2.is_done()

        pool.enqueue(task2)
        await asyncio.sleep(0.01)

        assert task0.is_done()
        assert task1.is_done()
        assert task2.is_done()

        pool.shutdown()

        response0 = await out_queue.get()
        response1 = await out_queue.get()
        response2 = await out_queue.get()

        assert response0 == task2.name
        assert response1 == task1.name
        assert response2 == task0.name

    asyncio.run(async_test())


def test_pool_timedout():
    async def async_test():
        out_queue = asyncio.Queue()
        event = asyncio.Event()

        task0 = Task(name="task-0", out_queue=out_queue)
        task0.set_dependent(event)

        pool = Pool(worker_count=4)
        pool.start()
        pool.enqueue(task0)

        try:
            await asyncio.wait_for(pool.wait(), timeout=0.1)
            timedout = False
        except asyncio.TimeoutError:
            timedout = True

        assert timedout

        event.set()

        try:
            await asyncio.wait_for(pool.wait(), timeout=0.1)
            timedout = False
        except asyncio.TimeoutError:
            timedout = True

        assert not timedout

    asyncio.run(async_test())
