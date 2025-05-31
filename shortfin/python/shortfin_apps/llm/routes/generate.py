# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from fastapi import APIRouter, Request

from shortfin.interop.fastapi import FastAPIResponder

from ..components.generate import ClientGenerateBatchProcess
from ..components.io_struct import GenerateReqInput
from ..components.service import LlmGenerateService

generation_router = APIRouter()


@generation_router.post("/generate")
@generation_router.put("/generate")
async def generate_request(gen_req: GenerateReqInput, request: Request):
    # app.state.services is populated by the ShortfinLlmLifecycleManager
    # see shortfin/python/shortfin_apps/llm/components/lifecycle.py
    service: LlmGenerateService = request.app.state.services["default"]
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(
        service, gen_req, responder, fiber=service.main_fiber
    ).launch()
    return await responder.response
