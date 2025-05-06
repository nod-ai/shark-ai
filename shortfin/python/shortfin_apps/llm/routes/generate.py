# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from fastapi import APIRouter, Request

from shortfin.interop.fastapi import FastAPIResponder

from ..components.io_struct import GenerateReqInput
from ..components.service_manager import LlmServiceManager

generation_router = APIRouter()
logger = logging.getLogger(__name__)


@generation_router.post("/generate")
@generation_router.put("/generate")
async def generate_request(gen_req: GenerateReqInput, request: Request):
    # app.state.service_manager is populated by the ShortfinLlmLifecycleManager
    # see shortfin/python/shortfin_apps/llm/components/lifecycle.py

    gen_req.post_init()
    responder = FastAPIResponder(request)
    responder.start_response()
    is_streaming: bool = gen_req.stream

    def response_handler(
        response: bytes | list[bytes], responder=responder, is_streaming=is_streaming
    ):
        """
        Converts a raw response from the service into a format suitable for
        FastAPIResponder.
        """
        try:
            if is_streaming:
                responder.stream_start()
                if isinstance(response, list):
                    for part in response:
                        responder.stream_part(part)
                else:
                    responder.stream_part(response)
            else:
                responder.send_response(response)
        finally:
            responder.ensure_response()

    service_manager: LlmServiceManager = request.app.state.service_manager
    await service_manager.send_request(gen_req, response_handler)
    response = await responder.response
    return response
