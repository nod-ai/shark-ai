# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import time
from fastapi import APIRouter, Request

from shortfin.interop.fastapi import FastAPIResponder

from ..components.generate import ClientGenerateBatchProcess
from ..components.io_struct import GenerateReqInput
from ..components.service import LlmGenerateService

generation_router = APIRouter()
logger = logging.getLogger(__name__)


@generation_router.post("/generate")
@generation_router.put("/generate")
async def generate_request(gen_req: GenerateReqInput, request: Request):
    # app.state.services is populated by the ShortfinLlmLifecycleManager
    # see shortfin/python/shortfin_apps/llm/components/lifecycle.py

    request.app.state.request_counter += 1
    service: LlmGenerateService = request.app.state.services[
        request.app.state.current_instance
    ]
    if not service.add_to_queue():
        request.app.state.current_instance = (
            request.app.state.current_instance + 1
        ) % len(request.app.state.services)
        service = request.app.state.services[request.app.state.current_instance]
        time.sleep(0.001)
    print(
        f"Generating with service: {request.app.state.current_instance} current queue size: {service.current_queue_size} max queue size: {service.max_queue_size}"
    )
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(service, gen_req, responder).launch()
    response = await responder.response
    service.remove_from_queue()
    return response
