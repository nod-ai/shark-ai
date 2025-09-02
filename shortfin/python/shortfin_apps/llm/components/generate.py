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
import traceback

from copy import deepcopy
from typing import List

import shortfin as sf
import threading

# TODO: Have a generic "Responder" interface vs just the concrete impl.
from shortfin.support.responder import AbstractResponder, ResponderErrorCodes
from shortfin_apps.llm.components.decoder.decoder import LlmDecoder, LogitsNormalization

from .config_struct import DecodeConfig
from .io_struct import (
    GenerateReqInput,
    GeneratedResponse,
    GenerateReqOutput,
    PromptResponse,
)
from .prefill_config import PrefillConfig
from .service import LlmGenerateService

from .tokenizer import Encoding

logger = logging.getLogger(__name__)


class GenerateItemProcess(sf.Process):
    def __init__(
        self,
        *,
        rid: int,
        unified_batcher,
        page_cache,
        input_text: str,
        input_token_ids: list[int],
        prefill_config: PrefillConfig,
        decode_config: DecodeConfig,
        fiber: sf.Fiber,
        use_native_impls: bool = False,
    ):
        super().__init__(fiber=fiber)
        self.rid = rid
        self.input_text = input_text
        self.input_token_ids = input_token_ids
        self.result_token_ids: list[int] = []
        self._prefill_config = prefill_config
        self.decode_config = decode_config
        self.cache = page_cache
        self.decoder = LlmDecoder(
            prefill_config=prefill_config,
            decode_config=decode_config,
            unified_batcher=unified_batcher,
            results_callback=self.results_callback,
            rid=self.rid,
            use_native_impls=use_native_impls,
        )

    def cancel(self):
        self.decoder.cancel()

    async def run(self):
        try:
            await self.decoder.run(input_ids=self.input_token_ids)
        except Exception:
            logger.error(traceback.format_exc())
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
        "gen_req",
        "lock",
        "unified_batcher",
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
        self.unified_batcher = self.service.unified_batcher
        self.complete_infeed = self.system.create_queue()
        self.active_processes = []
        self.cancelled = False
        self.lock = threading.Lock()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            for process in self.active_processes:
                process.cancel()

    def get_prefill_config(self) -> PrefillConfig:
        return PrefillConfig(
            has_prefill_position=self.service.model_params.has_prefill_position,
        )

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

    def validate_decode_config(self, responder, decode_config: DecodeConfig):
        has_softmax = decode_config.logits_normalization != LogitsNormalization.NONE
        has_temperature = (
            decode_config.temperature is not None and decode_config.temperature != 1.0
        )
        if has_softmax and has_temperature:
            responder.send_error(
                error_message=f"Temperature only supported for logits return {decode_config.temperature}.",
                code=ResponderErrorCodes.INVALID_REQUEST_ARGS,
                extra_fields={},
            )
            return False

        return True

    async def run(self):
        logger.debug("Started ClientBatchGenerateProcess: %r", self)

        prefill_config = self.get_prefill_config()
        decode_configs = self.get_decode_configs()

        input_ids = self.gen_req.input_ids
        is_pretokenized = input_ids is not None
        # TODO: We should send this to an executor and await the results.
        if is_pretokenized:
            input_batch = [input_ids] if self.gen_req.is_single else input_ids
        else:
            input_batch = self.tokenize()

        for config in decode_configs:
            if not self.validate_decode_config(self.responder, config):
                return

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
                gen_process = GenerateItemProcess(
                    unified_batcher=self.service.unified_batcher,
                    page_cache=self.service.page_cache,
                    rid=rid,
                    input_text=input_text,
                    input_token_ids=input_tokens,
                    prefill_config=prefill_config,
                    decode_config=decode_config,
                    fiber=fiber,
                    use_native_impls=self.service.server_params.use_native_impls,
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
        except Exception:
            logger.error(traceback.format_exc())
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
            self.responder.send_response(result_tokens)
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
