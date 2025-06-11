# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import inspect

from . import lib as sfl
import asyncio.futures as futures
import concurrent.futures

# Feature detect some versions where signatures changes.
if "context" in inspect.signature(asyncio.Task).parameters:
    # Python > 3.10
    _ASYNCIO_TASK_HAS_CONTEXT = True
else:
    _ASYNCIO_TASK_HAS_CONTEXT = False


class PyWorkerEventLoop(asyncio.AbstractEventLoop):
    def __init__(self, worker: sfl.local.Worker):
        self._worker = worker
        self._closed = False
        self._default_executor = None
        self._executor_shutdown_called = False

    def get_debug(self):
        # Requirement of asyncio.
        return False

    if _ASYNCIO_TASK_HAS_CONTEXT:

        def create_task(self, coro, *, name=None, context=None):
            return asyncio.Task(coro, loop=self, name=name, context=context)

    else:

        def create_task(self, coro, *, name=None):
            return asyncio.Task(coro, loop=self, name=name)

    def create_future(self):
        return asyncio.Future(loop=self)

    def time(self) -> float:
        return self._worker._now() / 1e9

    def call_soon_threadsafe(self, callback, *args, context=None) -> asyncio.Handle:
        def on_worker():
            asyncio.set_event_loop(self)
            return callback(*args)

        self._worker.call_threadsafe(on_worker)
        # TODO: Return future.

    def call_soon(self, callback, *args, context=None) -> asyncio.Handle:
        handle = _Handle(callback, args, self, context)
        self._worker.call(handle._sf_maybe_run)
        return handle

    def call_later(
        self, delay: float, callback, *args, context=None
    ) -> asyncio.TimerHandle:
        w = self._worker
        deadline = w._delay_to_deadline_ns(delay)
        handle = _TimerHandle(deadline / 1e9, callback, args, self, context)
        w.delay_call(deadline, handle._sf_maybe_run)
        return handle

    def call_at(self, when, callback, *args, context=None) -> asyncio.TimerHandle:
        w = self._worker
        deadline = int(when * 1e9)
        handle = _TimerHandle(when, callback, args, self, context)
        w.delay_call(deadline, handle._sf_maybe_run)
        return handle

    def call_exception_handler(self, context) -> None:
        # TODO: Should route this to the central exception handler. Should
        # also play with ergonomics of how the errors get reported in
        # various contexts and optimize.
        source_exception = context.get("exception")
        if isinstance(source_exception, BaseException):
            raise RuntimeError(
                f"Async exception on {self._worker}): {source_exception}"
            ).with_traceback(source_exception.__traceback__)
        else:
            raise RuntimeError(f"Async exception on {self._worker}: {context}")

    def run_in_executor(self, executor, func, *args):
        self._check_closed()
        if executor is None:
            executor = self._default_executor
            self._check_default_executor()
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix="shortfin_asyncio_bridge"
                )
                self._default_executor = executor
        return futures.wrap_future(executor.submit(func, *args), loop=self)

    def is_closed(self):
        return self._closed

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor_shutdown_called = True
        executor = self._default_executor
        if executor is not None:
            # Do not wait for the executor threads to finish.
            self._default_executor = None
            executor.shutdown(wait=False)

    def _timer_handle_cancelled(self, handle):
        # We don't do anything special: just skip it if it comes up.
        ...

    def _set_result_unless_cancelled(self, fut, result):
        if fut.cancelled():
            return
        fut.set_result(result)

    def _check_default_executor(self):
        if self._executor_shutdown_called:
            raise RuntimeError("Executor shutdown has already been called.")

    def _check_closed(self):
        if self._closed:
            raise RuntimeError("Event loop closed.")


class _Handle(asyncio.Handle):
    def _sf_maybe_run(self):
        if self.cancelled():
            return
        self._run()


class _TimerHandle(asyncio.TimerHandle):
    def _sf_maybe_run(self):
        if self.cancelled():
            return
        self._run()
