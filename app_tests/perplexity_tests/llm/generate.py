# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import logging
from typing import List, Optional

import torch
import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin_apps.llm.components.io_struct import GenerateReqInput
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase
from shortfin_apps.llm.components.service import GenerateService
from shortfin_apps.llm.components.tokenizer import Encoding

logger = logging.getLogger(__name__)


class GenerateItemProcess(sf.Process):
    """Process instantiated for each generation sequence.

    This process breaks the sequence into individual inference and sampling
    steps, submitting them to the batcher and marshaling incremental/final
    results.
    """

    def __init__(
        self,
        client: "ClientGenerateBatchProcess",
        gen_req: GenerateReqInput,
        index: int,
        input_token_ids: list[int],
        max_completion_tokens: int,
        eos_token_id: int,
        input_groundtruth: Optional[list[int]] = None,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_token_ids = input_token_ids  #
        self.result_token_ids: list[int] = []
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id
        self.input_groundtruth = input_groundtruth

    async def run(self):
        # TODO:
        exec = InferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_groundtruth[
                [0], [0], [0]
            ],  # TODO: pass all first tokens of the entire batch of input_groundtruth
            rid=self.gen_req.rid,
        )

        all_logits = []
        try:
            self.client.batcher.submit(exec)
            await exec.done

            # Prefill result.
            prefill_logits = exec.result_logits
            token = sfnp.argmax(prefill_logits)
            all_logits = prefill_logits

            # token_int = token.items[0]
            # self.append_token(token_int)

            self.append_token(
                self.input_groundtruth[[0], [0], [0]]
            )  # TODO: pass all first tokens of the entire batch of input_groundtruth

            # Decode loop.
            exec.start_position = len(self.input_token_ids) - 1
            exec.start_position = len(self.input_token_ids) - 1

            for i in range(self.max_completion_tokens):
                exec.reset(InferencePhase.DECODE)
                exec.input_token_ids.append(
                    self.input_groundtruth[[1], [1], [1]]
                )  # TODO: pass all consecutive tokens of the entire batch of input_groundtruth
                exec.start_position += 1
                self.client.batcher.submit(exec)
                await exec.done
                decode_logits = exec.result_logits
                all_logits = torch.cat(
                    (all_logits, decode_logits), 1
                )  # only torch package dependency in shortfin, find alternatives to concat w/o torch

                token = sfnp.argmax(decode_logits)
                self.append_token(
                    self.input_groundtruth[[1], [1], [1]]
                )  # TODO: pass all consecutive tokens of the entire batch of input_groundtruth
                # token_int = token.items[0]
                # self.append_token(token_int)
                # if token_int == self.eos_token_id:
                #     break
        finally:
            exec.free_cache_pages()

    def append_token(self, token: int):
        self.result_token_ids.append(token)
        self.client.stream_results(self)


class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization / Detokenization
    * Splitting the batch into GenerateItemProcesses
    * Streaming responses
    * Final responses
    """

    __slots__ = [
        "batcher",
        "complete_infeed",
        "gen_req",
        "responder",
        "tokenizer",
    ]

    def __init__(
        self,
        service: GenerateService,
        gen_req: GenerateReqInput,
        responder: FastAPIResponder,
    ):
        super().__init__(fiber=service.main_fiber)
        self.gen_req = gen_req
        self.responder = responder
        self.tokenizer = service.tokenizer
        self.batcher = service.batcher
        self.complete_infeed = self.system.create_queue()

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        streaming = self.gen_req.stream
        if streaming:
            self.responder.start_response()

        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            # TODO: We should send this to an executor and await the results.
            input_batch = self.tokenize()
            for index, input_tokens in enumerate(input_batch):
                gen_process = GenerateItemProcess(
                    self,
                    self.gen_req,
                    index,
                    input_tokens.ids,
                    max_completion_tokens=self.gen_req.sampling_params[
                        "max_completion_tokens"
                    ],
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)

            if streaming:
                logger.debug("Responding to streaming batch")
                self.responder.stream_part(b"data: [DONE]\n\n")
                self.responder.stream_part(None)
            else:
                logging.debug("Responding to one shot batch")
                out = io.BytesIO()
                result_texts = self.tokenizer.decode(
                    [p.result_token_ids for p in gen_processes]
                )
                for result_text in result_texts:
                    out.write(b"data: ")
                    out.write(result_text.encode())
                    out.write(b"\n\n")
                self.responder.send_response(out.getvalue())
        finally:
            self.responder.ensure_response()

    def stream_results(self, gen_process: GenerateItemProcess):
        if not self.gen_req.stream:
            return
        (result_text,) = self.tokenizer.decode([gen_process.result_token_ids])
        out = io.BytesIO()
        out.write(b"data: ")
        out.write(result_text.encode())
        out.write(b"\n\n")
        self.responder.stream_part(out.getvalue())

    def tokenize(self) -> list[Encoding]:
        gen_req = self.gen_req
        if gen_req.text is not None:
            if self.gen_req.is_single:
                texts = [self.gen_req.text]
                logger.debug("Encoding single request")
            else:
                texts = self.gen_req.text
                logger.debug("Encoding batch of %d", len(texts))
            encodings = self.tokenizer.encode(texts)
            logger.debug("Generated encodings: %r", encodings)
            return encodings
        else:
            raise NotImplementedError("GenerateReqInput.input_ids handling NYI")
