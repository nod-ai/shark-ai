# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import io
import json
import logging
from time import time
from typing import List

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .io_struct import GenerateReqInput
from .messages import LlmInferenceExecRequest, InferencePhase, LlmInferenceMetrics
from .service import GenerateService
from .tokenizer import Encoding

from .decode_strategy import (
    DecodeStrategy,
    DecodeStrategyConfig,
    BeamSearchDecodeStrategy,
    BeamSearchDecodeStrategyConfig,
    GreedyDecodeStrategy,
)

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
        n_beams: int,
        temperature: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.decode_strategy: DecodeStrategy = None
        self.n_beams = n_beams

        if gen_req.return_metrics:
            self.llm_inference_metrics = []

        if n_beams > 1:
            logger.info(f"Using `beam_search` decode strategy with {n_beams} beams")
            decode_strategy_config = BeamSearchDecodeStrategyConfig(
                batcher_callback=self.client.batcher.submit,
                streaming_callback=self.append_token,
                eos_token_id=eos_token_id,
                max_completion_tokens=max_completion_tokens,
                n_beams=n_beams,
                temperature=temperature,
                return_top_k=gen_req.return_top_k,
            )
            self.decode_strategy = BeamSearchDecodeStrategy(decode_strategy_config)
        else:
            logger.info(f"Using `greedy` decode strategy")
            if gen_req.return_top_k:
                gen_req.return_top_k = False
            decode_strategy_config = DecodeStrategyConfig(
                batcher_callback=self.client.batcher.submit,
                streaming_callback=self.append_token,
                eos_token_id=eos_token_id,
                max_completion_tokens=max_completion_tokens,
            )
            self.decode_strategy = GreedyDecodeStrategy(
                decode_strategy_config,
            )

        self.streamed_tokens_index = 0

    async def run(self):
        exec = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_token_ids,
            rid=self.gen_req.rid,
        )
        if self.gen_req.return_metrics:
            exec.llm_inference_metrics = LlmInferenceMetrics(
                prefill_times=[],
                decode_times=[],
                batcher_pending_times=[],
                start_time=time(),
            )
        try:
            self.client.batcher.submit(exec)
            await exec.done

            # Prefill result.
            token = sfnp.argmax(exec.result_logits)
            token_int = token.items[0]
            if self.gen_req.return_metrics:
                exec.llm_inference_metrics.set_ttft(time())

            self.append_token(token_int)
            # Decode loop.
            exec.start_position = len(self.input_token_ids) - 1
            exec.input_token_ids.append(token_int)
            exec.output_token_ids.append(token_int)
            await self.decode_strategy.decode(exec)
        finally:
            logger.info(f"Freeing cache pages: {exec.rid}")
            exec.free_cache_pages()

    def append_token(
        self,
        token: int | List[int],
        inference_metrics: (
            LlmInferenceMetrics | List[LlmInferenceMetrics] | None
        ) = None,
    ):
        if isinstance(token, list):
            self.result_token_ids = token
        else:
            self.result_token_ids.append(token)

        if isinstance(inference_metrics, list):
            self.llm_inference_metrics = inference_metrics
        elif inference_metrics is not None:
            self.llm_inference_metrics.append(inference_metrics)

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
        "n_beams",
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
        self.n_beams = service.server_params.n_beams

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)
        streaming = self.gen_req.stream
        if streaming:
            self.responder.start_response()

        try:
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            input_ids = self.gen_req.input_ids
            is_pretokenized = input_ids is not None
            # TODO: We should send this to an executor and await the results.
            if is_pretokenized:
                input_batch = [input_ids] if self.gen_req.is_single else input_ids
            else:
                input_batch = self.tokenize()
            for index, input_tokens in enumerate(input_batch):
                max_completion_tokens = (
                    self.gen_req.sampling_params["max_completion_tokens"]
                    if self.gen_req.is_single
                    else self.gen_req.sampling_params[index]["max_completion_tokens"]
                )
                temperature = (
                    self.gen_req.sampling_params["temperature"]
                    if self.gen_req.is_single
                    else self.gen_req.sampling_params[index]["temperature"]
                )
                requested_beams = (
                    self.gen_req.sampling_params.get("n_beams")
                    if isinstance(self.gen_req.sampling_params, dict)
                    else self.gen_req.sampling_params[index].get("n_beams")
                )
                n_beams = self.n_beams
                if requested_beams is not None:
                    n_beams = requested_beams

                # In case somebody requests top_k, with the server/request
                # set to `greedy`.
                if n_beams == 1 or n_beams is None:
                    self.gen_req.return_top_k = False
                gen_process = GenerateItemProcess(
                    self,
                    self.gen_req,
                    index,
                    input_tokens if is_pretokenized else input_tokens.ids,
                    max_completion_tokens=max_completion_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    n_beams=n_beams,
                    temperature=temperature,
                )
                gen_processes.append(gen_process)
                gen_process.launch()

            await asyncio.gather(*gen_processes)
            self.respond(gen_processes, streaming)
        except Exception as e:
            logger.exception(e)
        finally:
            self.responder.ensure_response()

    def respond(self, gen_processes: List[GenerateItemProcess], streaming: bool):
        if streaming:
            logger.debug("Responding to streaming batch")
            self.responder.stream_part(b"data: [DONE]\n\n")
            self.responder.stream_part(None)
        else:
            logging.debug("Responding to one shot batch")
            out = io.BytesIO()
            result_tokens = [p.result_token_ids for p in gen_processes]
            result_metrics = None
            if self.gen_req.return_metrics:
                result_metrics = []
                for p in gen_processes:
                    result_metrics.extend(p.llm_inference_metrics)
            if self.gen_req.return_input_ids:
                if self.gen_req.is_single:
                    result_tokens = result_tokens[0]
                    result_metrics = result_metrics[0]
                out.write(bytes(json.dumps(result_tokens), "utf-8"))
                if result_metrics is not None:
                    out.write(b"\n")
                    out.write(bytes(str(result_metrics), "utf-8"))
            else:
                if self.gen_req.return_top_k:
                    logger.info("Returning `topk` results.")
                    result_metrics = None
                    if self.gen_req.return_metrics:
                        result_metrics = [
                            p.llm_inference_metrics for p in gen_processes
                        ]
                    for batch_index, batch in enumerate(result_tokens):
                        result_texts = self.tokenizer.decode(batch)
                        for text_index, result_text in enumerate(result_texts):
                            out.write(b"data: ")
                            out.write(result_text.encode())
                            out.write(b"\n\n")
                            if result_metrics is not None:
                                out.write(b"\n")
                                out.write(
                                    bytes(
                                        str(result_metrics[batch_index][text_index]),
                                        "utf-8",
                                    )
                                )
                                out.write(b"\n\n")
                else:
                    logger.info("Returning result.")
                    result_texts = self.tokenizer.decode(result_tokens)
                    if self.gen_req.return_metrics:
                        result_metrics = []
                        for p in gen_processes:
                            result_metrics.extend(p.llm_inference_metrics)
                    for index, result_text in enumerate(result_texts):
                        out.write(b"data: ")
                        out.write(result_text.encode())
                        out.write(b"\n\n")
                        if self.gen_req.return_metrics:
                            out.write(bytes(str(result_metrics[index]), "utf-8"))
                            out.write(b"\n\n")
            self.responder.send_response(out.getvalue())

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
        self.responder.stream_part(out.getvalue())
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
