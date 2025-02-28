# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import base64
import logging
import json

from typing import (
    TypeVar,
    Union,
)

from PIL import Image

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
        self.result_image: Union[str, None] = None

    async def run(self):
        exec = SDXLInferenceExecRequest.from_batch(self.gen_req, self.index)
        self.client.batcher.submit(exec)
        await exec.done
        self.result_image = exec.result_image


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
                if each_process.result_image is None:
                    raise Exception(
                        f"Expected image result for batch {index_of_each_process} but got `None`"
                    )

                size_of_each_image = (
                    from_batch(self.gen_req.width, index_of_each_process),
                    from_batch(self.gen_req.height, index_of_each_process),
                )

                rgb_sequence_of_each_image = Base64CharacterEncodedByteSequence(
                    each_process.result_image
                )

                each_image = Image.frombytes(
                    mode="RGB",
                    size=size_of_each_image,
                    data=rgb_sequence_of_each_image.as_bytes,
                )

                png_images.append(png_from(each_image))

            response_body = {"images": png_images}
            response_body_in_json = json.dumps(response_body)
            self.responder.send_response(response_body_in_json)
        finally:
            self.responder.ensure_response()
