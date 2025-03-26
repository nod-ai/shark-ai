# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import time
from typing import Any
import functools

logger = logging.getLogger("shortfin-sd.metrics")


def measure(fn=None, type="exec", task=None, num_items=None, freq=1, label="items"):
    assert callable(fn) or fn is None

    def _decorator(func):
        @functools.wraps(func)
        async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            ret = await func(*args, **kwargs)
            duration = time.time() - start
            if type == "exec":
                batch_size = len(getattr(args[0], "exec_requests", []))
                log_duration_str(duration, task=task, batch_size=batch_size)
            if type == "throughput":
                if isinstance(num_items, str):
                    items = getattr(args[0].gen_req, num_items)
                else:
                    items = str(num_items)
                log_throughput(duration, items, freq, label)
            return ret

        return wrapped_fn_async

    return _decorator(fn) if callable(fn) else _decorator


def log_throughput(duration, num_items, freq, label) -> str:
    sps = str(float(num_items) / duration) * freq
    freq_str = "second" if freq == 1 else f"{freq} seconds"
    logger.info(f"THROUGHPUT: {sps} {label} per {freq_str}")


def log_duration_str(duration: float, task, batch_size=0) -> str:
    """Get human readable duration string from start time"""
    if batch_size > 0:
        task = f"{task} (batch size {batch_size})"
    duration_str = f"{round(duration * 1e3)}ms"
    logger.info(f"Completed {task} in {duration_str}")
