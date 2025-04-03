# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging

from typing import (
    TypeVar,
    Union,
)
from fastapi.responses import JSONResponse

from shortfin_apps.types.Base64CharacterEncodedByteSequence import (
    Base64CharacterEncodedByteSequence,
)
from shortfin_apps.utilities.image import png_from
from shortfin_apps.text_to_image.TextToImageInferenceOutput import (
    TextToImageInferenceOutput,
)

import shortfin as sf

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import FluxInferenceExecRequest
from .service import FluxGenerateService
from .metrics import measure

logger = logging.getLogger("shortfin-flux.generate")


class GenerateImageProcess(sf.Process):
    """Process instantiated for every image generation.

    This process breaks the sequence into individual inference and sampling
    steps, submitting them to the batcher and marshaling final
    results.

    Responsible for a single image.
    """

    def __init__(
        self,
        client: "ClientGenerateBatchProcess",
        gen_req: GenerateReqInput,
        index: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.output: Union[TextToImageInferenceOutput, None] = None

    async def run(self):
        exec = FluxInferenceExecRequest.from_batch(self.gen_req, self.index)
        self.client.batcher.submit(exec)
        await exec.done
        self.output = (
            TextToImageInferenceOutput(exec.response_image)
            if exec.response_image
            else None
        )


class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization
    * Random Latents Generation
    * Splitting the batch into GenerateImageProcesses
    * Streaming responses
    * Final responses
    """

    __slots__ = [
        "batcher",
        "complete_infeed",
        "gen_req",
        "responder",
    ]

    def __init__(
        self,
        service: FluxGenerateService,
        gen_req: GenerateReqInput,
        responder: FastAPIResponder,
    ):
        super().__init__(fiber=service.fibers[0])
        self.gen_req = gen_req
        self.responder = responder
        self.batcher = service.batcher
        self.complete_infeed = self.system.create_queue()

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes: list[GenerateImageProcess] = []
            for index in range(self.gen_req.num_output_images):
                gen_process = GenerateImageProcess(self, self.gen_req, index)
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)

            # TODO: stream image outputs
            logging.debug("Responding to one shot batch")

            png_images: list[Base64CharacterEncodedByteSequence] = []

            for index_of_each_process, each_process in enumerate(gen_processes):
                if each_process.output is None:
                    raise Exception(
                        f"Expected output for process {index_of_each_process} but got `None`"
                    )

                each_png_image = png_from(each_process.output.image)
                png_images.append(each_png_image)

            self.responder.send_response(
                JSONResponse(
                    content={
                        "images": png_images,
                    },
                    media_type="application/json",
                )
            )
        finally:
            self.responder.ensure_response()
