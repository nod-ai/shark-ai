# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import dataclasses
import io
import json
import logging

from copy import copy
from typing import List

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder
from fastapi.responses import JSONResponse
from fastapi import status

from .config_struct import DecodeConfig
from .io_struct import (
    GenerateReqInput,
    GeneratedResponse,
    GenerateReqOutput,
    PromptResponse,
)
from .messages import LlmInferenceExecRequest, InferencePhase
from .service import LlmGenerateService
from .token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
    build_token_selector,
    build_token_selector_config,
    is_multi_response,
    TokenSelectionStrategy,
)
from .tokenizer import Encoding

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
        input_text: str,
        input_token_ids: list[int],
        eos_token_id: int,
        decode_config: DecodeConfig,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_text = input_text
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.eos_token_id = eos_token_id
        self.decode_config = decode_config
        self.token_selector_config: TokenSelectionStrategyConfig = (
            build_token_selector_config(
                decode_config,
                prefill_batcher=self.client.prefill_batcher,
                decode_batcher=self.client.decode_batcher,
                results_callback=self.results_callback,
                eos_token_id=self.eos_token_id,
            )
        )
        self.token_selector: BaseTokenSelectionStrategy = build_token_selector(
            self.token_selector_config,
        )

        self.streamed_tokens_index = 0

    async def run(self):
        exec_req = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_token_ids,
            rid=self.gen_req.rid,
            decode_bs=self.decode_config.max_decode_batch_size,
            stream=self.gen_req.stream,
        )
        exec_req._cache = self.client.prefill_batcher.page_cache
        try:
            # Prefill result.
            await self.token_selector.prefill(exec_req)

            # Decode loop.
            await self.token_selector.decode(exec_req)
        finally:
            exec_req.free_cache_pages()

    def results_callback(self, result: int | list[list[int]]):
        if not self.gen_req.stream and is_multi_response(
            self.decode_config.token_selection_strategy
        ):
            # TODO: Streaming is not supported for multiple responses
            self.result_token_ids = result
            logger.info(f"Received multi-response: {self.result_token_ids}")
            return

        self._append_token(result)

    def _append_token(self, token: int):
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
        "complete_infeed",
        "decode_batcher",
        "gen_req",
        "prefill_batcher",
        # "responder",
        "response_handler",
        "tokenizer",
        "decode_config",
        "service",
    ]

    def __init__(
        self,
        service: LlmGenerateService,
        gen_req: GenerateReqInput,
        # responder: FastAPIResponder,
        response_handler: callable,
        # takes a single arg which is bytes/None, or a list of bytes/None
        # for streaming
        fiber: sf.Fiber | None = None,
    ):
        super().__init__(
            fiber=service.fiber_pool.fibers[0].fiber if fiber is None else fiber
        )
        self.gen_req = gen_req
        # self.responder = responder
        self.response_handler = response_handler
        self.tokenizer = service.tokenizer
        self.prefill_batcher = service.prefill_batcher
        self.decode_batcher = service.decode_batcher
        self.complete_infeed = self.system.create_queue()
        self.service = service
        self.decode_config = service.server_params.decode_config
        self.decode_config.token_selection_strategy = (
            TokenSelectionStrategy.MULTI_GREEDY
            if self.gen_req.sampling_params.b_of_n > 1
            else TokenSelectionStrategy.GREEDY
        )

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)

        # Try to add request to queue
        if not self.service.add_to_queue():
            error_response = JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Server queue is full. Please try again later.",
                    "code": "QUEUE_FULL",
                    "current_size": self.service.current_queue_size,
                    "max_size": self.service.max_queue_size,
                },
            )
            self.responder.send_response(error_response)
            self.responder.ensure_response()
            return

        # try:
        streaming = self.gen_req.stream
        # self.responder.start_response()
        # if streaming:
        #     self.responder.stream_start()

        # Launch all individual generate processes and wait for them to finish.
        gen_processes = []
        input_ids = self.gen_req.input_ids
        is_pretokenized = input_ids is not None

        if is_pretokenized:
            input_batch = [input_ids] if self.gen_req.is_single else input_ids
        else:
            input_batch = self.tokenize()
        for index, input_tokens in enumerate(input_batch):
            decode_config = copy(self.decode_config)
            decode_config.update_from_sampling_params(
                self.gen_req.sampling_params
                if self.gen_req.is_single
                else self.gen_req.sampling_params[index]
            )
            gen_process = GenerateItemProcess(
                self,
                self.gen_req,
                index,
                self.gen_req.text
                if self.gen_req.is_single
                else self.gen_req.text[index],
                input_tokens if is_pretokenized else input_tokens.ids,
                eos_token_id=self.tokenizer.eos_token_id,
                decode_config=decode_config,
            )
            gen_processes.append(gen_process)
            gen_process.launch()

        await asyncio.gather(*gen_processes)
        self.generate_response(gen_processes, streaming)
        # finally:
        #     self.responder.ensure_response()

    def generate_response(
        self, gen_processes: List[GenerateItemProcess], streaming: bool
    ):
        if streaming:
            logger.info("Responding to streaming batch")
            # self.responder.stream_part(b"data: [DONE]\n\n")
            # self.responder.stream_part(None)
            self.response_handler([b"data: [DONE]\n\n", None])
            return

        logging.debug("Responding to one shot batch")
        result_tokens = [p.result_token_ids for p in gen_processes]
        if self.gen_req.return_input_ids:
            if self.gen_req.is_single:
                result_tokens = result_tokens[0]
            out = io.BytesIO()
            out.write(bytes(json.dumps(result_tokens), "utf-8"))
            # self.responder.send_response(out.getvalue())
            self.response_handler(out.getvalue())
            return

        response_map = {}

        for p in gen_processes:
            response_map[p.input_text] = []

        for p in gen_processes:
            token_ids = p.result_token_ids

            if not is_multi_response(self.decode_config.token_selection_strategy):
                token_ids = [token_ids]

            decoded = self.tokenizer.decode(token_ids)
            rs = [GeneratedResponse(d) for d in decoded]
            response_map[p.input_text] += rs

        responses = []
        for k in response_map:
            r = PromptResponse(prompt=k, responses=response_map[k])
            r = dataclasses.asdict(r)
            responses.append(r)

        response = GenerateReqOutput(responses=responses)
        response = dataclasses.asdict(response)
        response = json.dumps(response)
        out = io.BytesIO()
        out.write(response.encode())
        # self.responder.send_response(out.getvalue())
        self.response_handler(out.getvalue())

    def _respond_multi_responses(
        self, result_token_ids: List[List[int]], out: io.BytesIO
    ) -> io.BytesIO:
        logger.debug("Responding to multi-beam request")

        # Parse each request in batch
        for token_ids in result_token_ids:
            result_texts = self.tokenizer.decode(token_ids)
            for result_text in result_texts:
                out.write(b"data: ")
                out.write(result_text.encode())
                out.write(b"\n\n")

        return out

    def stream_results(self, gen_process: GenerateItemProcess):
        if not self.gen_req.stream:
            return
        out = io.BytesIO()
        result_tokens = gen_process.result_token_ids[
            gen_process.streamed_tokens_index :
        ]
        rid = (
            gen_process.gen_req.rid
            if gen_process.gen_req.is_single
            else gen_process.gen_req.rid[gen_process.index]
        )
        if not self.gen_req.return_input_ids:
            (result_text,) = self.tokenizer.decode([result_tokens])
            out.write(f"data({rid}): ".encode())
            out.write(result_text.encode())
            out.write(b"\n\n")
        else:
            out.write(f"data({rid}): ".encode())
            out.write(str(result_tokens[0]).encode())
            out.write(b"\n\n")
        # self.responder.stream_part(out.getvalue())
        self.response_handler(out.getvalue())
        gen_process.streamed_tokens_index += len(result_tokens)

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
            raise ValueError("Cannot tokenize 'None' value")
