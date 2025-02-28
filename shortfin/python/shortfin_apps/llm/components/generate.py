# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import copy
import io
import json
import logging

import shortfin as sf
import shortfin.array as sfnp

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.interop.fastapi import FastAPIResponder

from .beam_manager import BeamManager
from .io_struct import GenerateReqInput
from .messages import InferenceExecRequest, InferencePhase
from .service import GenerateService
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
        input_token_ids: list[int],
        max_completion_tokens: int,
        eos_token_id: int,
        n_beams: int,
    ):
        super().__init__(fiber=client.fiber)
        self.client = client
        self.gen_req = gen_req
        self.index = index
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id
        self.n_beams = n_beams
        self.beam_manager = BeamManager(n_beams)

        self.streamed_tokens_index = 0

    async def run(self):
        exec = InferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_token_ids,
            rid=self.gen_req.rid,
        )
        try:
            self.client.batcher.submit(exec)
            await exec.done

            # Prefill result.
            token = sfnp.argmax(exec.result_logits)
            token_int = token.items[0]

            self.append_token(token_int)
            # Decode loop.
            exec.start_position = len(self.input_token_ids) - 1
            exec.input_token_ids.append(token_int)
            if self.n_beams > 1:
                await self.beam_search_decode_loop(exec)
            else:
                await self.greedy_decode_loop(exec)
        finally:
            logger.info(f"Freeing cache pages: {exec.rid}")
            exec.free_cache_pages()

    async def greedy_decode_loop(self, exec_req: InferenceExecRequest):
        for _ in range(self.max_completion_tokens):
            exec_req.reset(InferencePhase.DECODE)
            self.client.batcher.submit(exec_req)
            await exec_req.done
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]
            self.append_token(token_int)
            if token_int == self.eos_token_id:
                break
            exec_req.input_token_ids.append(token_int)
            exec_req.start_position += 1

    async def beam_search_decode_loop(self, exec_req: InferenceExecRequest):
        n_beams = self.n_beams
        decode_reqs = [exec_req]
        # First, we need to replicate our exec_req,
        # such that len(decode_reqs) == self.n_beams
        for _ in range(n_beams - 1):
            decode_req = exec_req.replicate_self()
            decode_reqs.append(decode_req)

        self.beam_manager.create_beam(decode_reqs)
        beam_group_id = exec_req.beam_group_id
        beam_group = self.beam_manager.beam_map[beam_group_id]
        for _ in range(self.max_completion_tokens):
            if len(beam_group.completed_reqs) == self.n_beams:
                break

            # Submit all decode requests to the batcher from this beam
            for exec in beam_group.exec_reqs:
                if exec in beam_group.completed_reqs:
                    continue
                exec.reset(InferencePhase.DECODE)
                self.client.batcher.submit(exec)

            # Wait for all beams to finish
            await beam_group.wait()
            beam_group.process_beams(self.eos_token_id)

        if self.gen_req.return_top_k:
            reqs = beam_group.completed_reqs
            for req in beam_group.exec_reqs:
                reqs.add(req)
            results = [req.input_token_ids for req in reqs]
            self.result_token_ids = results
            self.client.stream_results(self)
            self.beam_manager.delete_beam(beam_group_id)
            return

        selected_req = beam_group.find_top_beam()
        self.result_token_ids = selected_req.input_token_ids
        self.client.stream_results(self)
        self.beam_manager.delete_beam(beam_group_id)

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
        self.n_beams = service.model_params.n_beams

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
                gen_process = GenerateItemProcess(
                    self,
                    self.gen_req,
                    index,
                    input_tokens if is_pretokenized else input_tokens.ids,
                    max_completion_tokens=max_completion_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    n_beams=self.n_beams,
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
                result_tokens = [p.result_token_ids for p in gen_processes]
                if self.gen_req.return_input_ids:
                    if self.gen_req.is_single:
                        result_tokens = result_tokens[0]
                    out.write(bytes(json.dumps(result_tokens), "utf-8"))
                else:
                    if self.gen_req.return_top_k:
                        for batch in result_tokens:
                            result_texts = self.tokenizer.decode(batch)
                            for result_text in result_texts:
                                out.write(b"data: ")
                                out.write(result_text.encode())
                                out.write(b"\n\n")
                    else:
                        result_texts = self.tokenizer.decode(result_tokens)
                        for result_text in result_texts:
                            out.write(b"data: ")
                            out.write(result_text.encode())
                            out.write(b"\n\n")
                self.responder.send_response(out.getvalue())
        except Exception as e:
            logger.error(e)
        finally:
            self.responder.ensure_response()

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
