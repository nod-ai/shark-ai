# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Implements a context manager that configures a shortfin llm server from a namespace mirroring server.py's commandline args, and exposes a context manager interface such that we can do:

```python
def lifecycle(app: FastApi):
    with lifecycle_manager(args) as man:
        yield
```
"""

from .service_manager import (
    LlmServiceManager,
    LlmMultiProcessServiceManager,
    LlmSingleProcessServiceManager,
)

from contextlib import asynccontextmanager
from fastapi import FastAPI


def get_eos_from_tokenizer_config(json_path):
    import json

    with open(json_path, "rt") as f:
        json_text = f.read()
    config = json.loads(json_text)
    return config["eos_token"]


class ShortfinLlmLifecycleManager:
    """
    Manages the lifecycle of a shortfin llm server, including config loading and parameter setup.

    There are generally two ways to use this.

    To start a full shortfin server, use the context manager or the fastapi_lifespan method.

    To initialize a shortfin server but not start it, use the constructor, then manipulate the services and sysman attributes directly.
    """

    def __init__(self, args):
        self.service_manager: LlmServiceManager = (
            LlmSingleProcessServiceManager(args)
            if args.in_process
            else LlmMultiProcessServiceManager(args)
        )

    def __enter__(self):
        self.service_manager.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.service_manager.shutdown()
        return False

    @asynccontextmanager
    async def fastapi_lifespan(self, app: FastAPI):
        """
        Context manager for FastAPI lifespan events.

        Initializes the system manager and services when the app starts, and shuts them down when the app stops.
        Also provides the services via app.state, which can be accessed from route handlers via
        request.app.state.services.

        Implements API described in https://fastapi.tiangolo.com/advanced/events/#lifespan

        See `server.py` for a usage example.
        """
        with self:
            app.state.service_manager = self.service_manager
            yield
