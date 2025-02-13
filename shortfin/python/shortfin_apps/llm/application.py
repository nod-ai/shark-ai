# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from fastapi import FastAPI

from .lifecycle_hooks import lifespan
from .routes import application_router, generation_router


def add_routes(app: FastAPI):
    app.include_router(application_router)
    app.include_router(generation_router)
    return app


def get_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app = add_routes(app)
    return app
