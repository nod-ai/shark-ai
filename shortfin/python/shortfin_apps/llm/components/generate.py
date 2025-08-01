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

from copy import deepcopy
from typing import List, Tuple

import shortfin as sf
import threading

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.support.responder import AbstractResponder, ResponderErrorCodes
from shortfin_apps.llm.components.decoder.decoder import LlmDecoder

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
    TokenSelector,
    TokenSelectionStrategyConfig,
    build_token_selector_config,
    is_multi_response,
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
        *,
        rid: int,
        prefill_batcher,
        decode_batcher,
        page_cache,
        input_text: str,
        input_token_ids: list[int],
        decode_config: DecodeConfig,
        fiber: sf.Fiber,
    ):
        super().__init__(fiber=fiber)
        self.rid = rid
        self.input_text = input_text
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.decode_config = decode_config
        self.cache = page_cache
        self.token_selector_config: TokenSelectionStrategyConfig = (
            build_token_selector_config(
                decode_config,
                prefill_batcher=prefill_batcher,
                decode_batcher=decode_batcher,
                results_callback=self.results_callback,
            )
        )
        self.token_selector: TokenSelector = TokenSelector(
            token_selection_strategy_config=self.token_selector_config,
        )

    def cancel(self):
        self.token_selector.cancel()

    async def run(self):
        exec_req = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=self.input_token_ids,
            rid=self.rid,
        )
        exec_req._cache = self.cache
        try:
            # Prefill result.
            await self.token_selector.prefill(exec_req)
            # Decode loop.
            await self.token_selector.decode(exec_req)
        finally:
            exec_req.free_cache_pages()

    def results_callback(self, result: List[List[int]]):
        self.result_token_ids = result


class NewGenerateItemProcess(sf.Process):
    def __init__(
        self,
        *,
        rid: int,
        prefill_batcher,
        decode_batcher,
        page_cache,
        input_text: str,
        input_token_ids: list[int],
        decode_config: DecodeConfig,
        fiber: sf.Fiber,
        use_native_impls: bool = False,
    ):
        super().__init__(fiber=fiber)
        self.rid = rid
        self.input_text = input_text
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self.decode_config = decode_config
        self.cache = page_cache
        self.decoder = LlmDecoder(
            decode_config,
            prefill_batcher=prefill_batcher,
            decode_batcher=decode_batcher,
            results_callback=self.results_callback,
            rid=self.rid,
            use_native_impls=use_native_impls,
        )

    def cancel(self):
        self.decoder.cancel()

    async def run(self):
        try:
            await self.decoder.run(input_ids=self.input_token_ids)
        finally:
            self.decoder.release()

    def results_callback(self, result: list[list[int]]):
        self.result_token_ids = result


class ClientGenerateBatchProcess(sf.Process):
    """Process instantiated for handling a batch from a client.

    This takes care of several responsibilities:

    * Tokenization / Detokenization
    """

    __slots__ = [
        "active_processes",
        "cancelled",
        "complete_infeed",
        "decode_batcher",
        "gen_req",
        "lock",
        "prefill_batcher",
        "responder",
        "tokenizer",
        "decode_config",
        "service",
    ]

    def __init__(
        self,
        service: LlmGenerateService,
        gen_req: GenerateReqInput,
        responder: AbstractResponder,
        fiber: sf.Fiber,
    ):
        super().__init__(fiber=fiber)
        self.service = service
        self.gen_req = gen_req
        self.responder = responder
        self.tokenizer = service.tokenizer
        self.prefill_batcher = service.prefill_batcher
        self.decode_batcher = service.decode_batcher
        self.complete_infeed = self.system.create_queue()
        self.active_processes = []
        self.cancelled = False
        self.lock = threading.Lock()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            for process in self.active_processes:
                process.cancel()

    def get_decode_configs(self) -> List[DecodeConfig]:
        """Calculate the total number of beams requested in the generation request."""
        gen_req = self.gen_req
        decode_configs = []

        sampling_params = (
            [gen_req.sampling_params] if gen_req.is_single else gen_req.sampling_params
        )

        for sampling_param in sampling_params:
            decode_config = deepcopy(self.service.server_params.decode_config)
            decode_config.eos_token_id = self.tokenizer.eos_token_id
            decode_config.update_from_sampling_params(sampling_param)
            decode_configs.append(decode_config)

        return decode_configs

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)

        decode_configs = self.get_decode_configs()

        input_ids = self.gen_req.input_ids
        is_pretokenized = input_ids is not None
        # TODO: We should send this to an executor and await the results.
        if is_pretokenized:
            input_batch = [input_ids] if self.gen_req.is_single else input_ids
        else:
            input_batch = self.tokenize()

        # Try to add request to queue
        # TODO(@zphoenixrises): Add load testing and integration tests for this.
        run_request = self.service.queue_manager.add_to_queue(
            decode_configs=decode_configs,
            input_batch=input_batch,
            is_pretokenized=is_pretokenized,
            responder=self.responder,
        )
        if run_request is None:
            return

        try:
            indices = []
            # Launch all individual generate processes and wait for them to finish.
            gen_processes = []
            for index, input_tokens in enumerate(input_batch):
                decode_config = decode_configs[index]
                input_text = (
                    self.gen_req.text[index]
                    if not is_pretokenized and not self.gen_req.is_single
                    else self.gen_req.text
                )

                idx, fiber = await self.service.main_fiber_pool.get()
                indices.append(idx)

                rid = (
                    self.gen_req.rid
                    if self.gen_req.is_single
                    else self.gen_req.rid[index]
                )

                input_tokens = input_tokens if is_pretokenized else input_tokens.ids
                if self.service.server_params.use_new_decoder:
                    gen_process = NewGenerateItemProcess(
                        prefill_batcher=self.service.prefill_batcher,
                        decode_batcher=self.service.decode_batcher,
                        page_cache=self.service.page_cache,
                        rid=rid,
                        input_text=input_text,
                        input_token_ids=input_tokens,
                        decode_config=decode_config,
                        fiber=fiber,
                        use_native_impls=self.service.server_params.use_native_impls,
                    )
                else:
                    gen_process = GenerateItemProcess(
                        prefill_batcher=self.service.prefill_batcher,
                        decode_batcher=self.service.decode_batcher,
                        page_cache=self.service.page_cache,
                        rid=rid,
                        input_text=input_text,
                        input_token_ids=input_tokens,
                        decode_config=decode_config,
                        fiber=fiber,
                    )

                gen_processes.append(gen_process)
                gen_process.launch()

            # Track the active processes and cancel as necessary:
            with self.lock:
                if self.cancelled:
                    for p in gen_processes:
                        p.cancel()
                self.active_processes = gen_processes

            await asyncio.gather(*gen_processes)
            if self.cancelled:
                self.responder.send_error(
                    error_message="Request cancelled",
                    code=ResponderErrorCodes.CANCELLED,
                    extra_fields={},
                )
            else:
                self.generate_response(gen_processes)
        finally:
            self.service.main_fiber_pool.return_fiber(indices)
            self.responder.ensure_response()
            self.service.queue_manager.remove_from_queue(run_request)

    def generate_response(
        self,
        gen_processes: List[GenerateItemProcess],
    ):
        logging.debug("Responding to one shot batch")
        result_tokens = [p.result_token_ids for p in gen_processes]
        if self.gen_req.return_input_ids:
            if self.gen_req.is_single:
                result_tokens = result_tokens[0]
            out = io.BytesIO()
            out.write(bytes(json.dumps(result_tokens), "utf-8"))
            self.responder.send_response(out.getvalue())
            return

        response_map = {p.input_text: [] for p in gen_processes}

        for p in gen_processes:
            decoded = self.tokenizer.decode(p.result_token_ids)
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
        self.responder.send_response(out.getvalue())

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
