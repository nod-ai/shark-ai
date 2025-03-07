# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import json

from typing import (
    TypeVar,
    Union,
)

from shortfin_apps.types.Base64CharacterEncodedByteSequence import (
    Base64CharacterEncodedByteSequence,
)

from shortfin_apps.utilities.image import png_from

import shortfin as sf

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import SDXLInferenceExecRequest
from .service import SDXLGenerateService
from .metrics import measure
from .TextToImageInferenceOutput import TextToImageInferenceOutput

logger = logging.getLogger("shortfin-sd.generate")


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
        exec = SDXLInferenceExecRequest.from_batch(self.gen_req, self.index)
        self.client.batcher.submit(exec)
        await exec.done

        self.output = (
            TextToImageInferenceOutput(exec.response_image)
            if exec.response_image
            else None
        )


Item = TypeVar("Item")


def from_batch(
    given_subject: list[Item] | Item | None,
    given_batch_index,
) -> Item:
    if given_subject is None:
        raise Exception("Expected an item or batch of items but got `None`")

    if not isinstance(given_subject, list):
        return given_subject

    # some args are broadcasted to each prompt, hence overriding index for single-item entries
    if len(given_subject) == 1:
        return given_subject[0]

    return given_subject[given_batch_index]


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
        service: SDXLGenerateService,
        gen_req: GenerateReqInput,
        responder: FastAPIResponder,
    ):
        super().__init__(fiber=service.meta_fibers[0].fiber)
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

            response_body = {"images": png_images}
            response_body_in_json = json.dumps(response_body)
            self.responder.send_response(response_body_in_json)
        finally:
            self.responder.ensure_response()
