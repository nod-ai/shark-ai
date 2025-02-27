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


from .config_struct import ModelParams, ServerParams
from .manager import LlmSystemManager
from .service import LlmGenerateService
from .tokenizer import Tokenizer
from typing import TYPE_CHECKING
from fastapi import FastAPI


from contextlib import asynccontextmanager
import logging


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
        # Load server configuration with priority: command line > config file > defaults
        server_params = ServerParams.load(
            args.server_config if hasattr(args, "server_config") else None
        )
        server_params.update_from_args(args)

        # Setup system (configure devices, etc).
        sysman = LlmSystemManager(
            device=args.device,
            device_ids=server_params.device_ids,
            async_allocs=server_params.amdgpu_async_allocations,
            amdgpu_allocators=server_params.amdgpu_allocators,
        )

        # Setup each service we are hosting.
        eos_token = get_eos_from_tokenizer_config(args.tokenizer_config_json)
        tokenizer = Tokenizer.from_tokenizer_json_file(
            args.tokenizer_json, eos_token=eos_token
        )
        model_params = ModelParams.load_json(args.model_config)
        service = LlmGenerateService(
            name="default",
            sysman=sysman,
            tokenizer=tokenizer,
            model_params=model_params,
            server_params=server_params,
            program_isolation=server_params.program_isolation,
        )
        service.load_inference_module(args.vmfb)
        service.load_inference_parameters(*args.parameters, parameter_scope="model")
        self.sysman = sysman
        self.services = {"default": service}

    def __enter__(self):
        self.sysman.start()
        for service_name, service in self.services.items():
            logging.info("Initializing service '%s': %r", service_name, service)
            service.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for service_name, service in self.services.items():
            logging.info("Shutting down service '%s'", service_name)
            service.shutdown()
        self.sysman.shutdown()
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
            app.state.services = self.services
            yield
