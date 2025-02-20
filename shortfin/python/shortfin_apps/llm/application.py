# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Uses shortfin_apps.llm.components.lifecycle to configure a FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import application_router, generation_router
from fastapi import FastAPI


def add_routes(app: FastAPI):
    app.include_router(application_router)
    app.include_router(generation_router)
    return app


def add_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def get_app(lifespan) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app = add_routes(app)
    app = add_middleware(app)
    return app
