# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import asyncio
import itertools
import numpy as np
import threading
import math

from shortfin_apps.llm.components.kvcache.page_pool import PagePool
from shortfin_apps.llm.components.decode_config import (
    DecodeConfig,
    LogitsNormalization,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from typing import Callable, List, Optional, Tuple, Union

from _shortfin import lib as _sfl
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    CacheAllocationFailure,
)

logger = logging.getLogger(__name__)


def _convert_to_cpp_decode_config(py_config: DecodeConfig):
    cpp_config = _sfl.llm.DecodeConfig()
    cpp_config.eos_token_id = py_config.eos_token_id
    cpp_config.num_beams = py_config.num_beams
    cpp_config.temperature = py_config.temperature
    cpp_config.use_beam_search = py_config.num_beams > 1
    cpp_config.max_completion_tokens = py_config.max_completion_tokens

    # Convert LogitsNormalization enum
    cpp_config.logits_normalization = {
        LogitsNormalization.NONE: _sfl.llm.LogitsNormalization.NONE,
        LogitsNormalization.SOFTMAX: _sfl.llm.LogitsNormalization.SOFTMAX,
        LogitsNormalization.LOG_SOFTMAX: _sfl.llm.LogitsNormalization.LOG_SOFTMAX,
    }[py_config.logits_normalization]

    cpp_config.top_k = py_config.top_k if py_config.top_k is not None else -1
    cpp_config.top_p = py_config.top_p if py_config.top_p is not None else -1.0

    return cpp_config


def combine_scores_null(
    step: np.ndarray, old_score: np.ndarray, norm: float, config: DecodeConfig
):
    if config.temperature is not None:
        step = step / config.temperature

    step = step - np.log(np.sum(np.exp(step.astype(float))))
    new_score = old_score + step
    new_score = new_score - norm
    return new_score


def combine_scores_softmax(
    step: np.ndarray, old_score: np.ndarray, norm: float, config: DecodeConfig
):
    new_score = old_score * step
    new_score = new_score / max(norm, 0.1)
    return new_score


def combine_scores_log_softmax(
    step: np.ndarray, old_score: np.ndarray, norm: float, config: DecodeConfig
):
    new_score = old_score + step
    new_score = new_score - norm
    return new_score


_score_functions = {
    LogitsNormalization.NONE: combine_scores_null,
    LogitsNormalization.SOFTMAX: combine_scores_softmax,
    LogitsNormalization.LOG_SOFTMAX: combine_scores_log_softmax,
}


def select_greedy(scores: np.ndarray, decode_config: DecodeConfig):
    assert len(scores.shape) == 2
    scores = scores.flatten()
    argmax = np.argmax(scores)
    argmax = np.array([argmax])

    return argmax, scores[argmax]


def select_topk(scores: np.ndarray, decode_config: DecodeConfig):
    assert len(scores.shape) == 2
    scores = scores.flatten()
    num_select = decode_config.num_beams
    if num_select < scores.shape[0]:
        token = np.argpartition(scores, -num_select)
        token = np.flip(token[-num_select:])
    else:
        token = np.arange(scores.shape[0])

    return token, scores[token]


class PageManager:
    def __init__(
        self,
        page_pool: PagePool,
        initial_pages: List[int],
        initial_length: int,
        tokens_per_page: int,
    ):
        self._page_pool = page_pool
        self._allocated_pages = []
        self._allocated_page_ids = []
        self._free_pages = []
        self._beam_page_ids = [[]]

        self._tokens_per_page = tokens_per_page
        self._allocation_block_size = 8

        self._shared_pages = initial_pages
        self._position = initial_length

        if self._position % self._tokens_per_page > 0:
            self._beam_page_ids[0].append(self._shared_pages[-1])
            self._shared_pages.pop()

    def allocate(self, count):
        if count > len(self._free_pages):
            acquire_count = max(count, self._allocation_block_size)
            acquired = self._page_pool.acquire_free_pages(acquire_count)
            self._allocated_pages.extend(acquired)
            self._free_pages.extend([p.index for p in acquired])

        allocation = self._free_pages[:count]
        self._free_pages = self._free_pages[count:]
        return allocation

    def step_pages(self, select):
        if len(select) == 0:
            return

        new_page = (self._position % self._tokens_per_page) == 0
        new_beam_page_ids = [[p for p in self._beam_page_ids[b]] for b in select]

        old_pages = set(itertools.chain.from_iterable(self._beam_page_ids))
        new_pages = set(itertools.chain.from_iterable(new_beam_page_ids))

        free_pages = old_pages - new_pages
        self._free_pages.extend(free_pages)

        if new_page:
            for beam, page in zip(
                new_beam_page_ids, self.allocate(len(new_beam_page_ids))
            ):
                beam.append(page)
        else:
            used = set()
            for beam in new_beam_page_ids:
                if len(beam) > 0:
                    if beam[-1] in used:
                        new_page = self.allocate(1)[0]
                        self._page_pool.copy_page_index(beam[-1], new_page)
                        beam[-1] = new_page
                    used.add(beam[-1])

        # Check if the pages a shared between all queries:
        if len(new_beam_page_ids[0]) > 0:
            first_page = new_beam_page_ids[0][0]
            if all(first_page == b[0] for b in new_beam_page_ids):
                self._shared_pages.append(first_page)
                new_beam_page_ids = [b[1:] for b in new_beam_page_ids]

        self._beam_page_ids = new_beam_page_ids
        self._position += 1
        return [self._shared_pages + b for b in new_beam_page_ids]

    def release_pages(self):
        self._page_pool.free_pages(self._allocated_pages)
        self._allocated_pages = []


class TokenSelector:
    def __init__(self, decode_config: DecodeConfig):
        self._selected_tokens: List[List[int]] = []
        self._selected_beams: List[List[int]] = []
        self._scores: List[float] = [0.0]
        self._completed: List[Tuple[int, int]] = []

        self._decode_config = decode_config
        self._eos_token_id = self._decode_config.eos_token_id
        self._hypothesis = self._decode_config.num_beams

        self._select_function = None
        self._select_function = (
            select_topk if decode_config.num_beams > 1 else select_greedy
        )

        self._score_function = _score_functions[decode_config.logits_normalization]

    def _select(self, logits: List[np.ndarray], indices: List[Optional[np.ndarray]]):
        # Setup next steps:
        step = len(self._selected_beams)
        max_score = max(self._scores)

        logits = [
            self._score_function(np.asarray(l), s, max_score, self._decode_config)
            for l, s in zip(logits, self._scores)
        ]

        logits = np.concatenate(logits, axis=1)[0]
        token_options = logits.shape[-1]
        tokens, scores = self._select_function(logits, self._decode_config)

        if indices[0] is not None:
            indices = [np.asarray(i) for i in indices]
            indices = np.concatenate(indices, axis=1)[0]
            beams = tokens // token_options
            tokens = np.take(indices, tokens)
        else:
            beams = tokens // token_options
            tokens = tokens % token_options

        # Filter out eos cases
        eos = self._eos_token_id
        next_tokens = [token for token in tokens if token != eos]
        next_beams = [beam for token, beam in zip(tokens, beams) if token != eos]
        next_scores = [score for token, score in zip(tokens, scores) if token != eos]
        next_completed = [
            (beam, step) for token, beam in zip(tokens, beams) if token == eos
        ]

        self._completed.extend(next_completed)
        self._selected_beams.append(next_beams)
        self._selected_tokens.append(next_tokens)
        self._scores = next_scores

        return next_beams, next_tokens

    def step(self, logits: list[np.ndarray], indices: list[Optional[np.ndarray]]):
        beams, tokens = self._select(logits, indices)

        return beams, tokens

    def done(self):
        return len(self._completed) >= self._hypothesis

    def _build_response(self, beam, end_step):
        tokens = []
        for step in range(end_step - 1, -1, -1):
            token = self._selected_tokens[step][beam]
            beam = self._selected_beams[step][beam]
            tokens.append(token)
        tokens.reverse()
        return tokens

    def results(self):
        results = []
        for completed in self._completed:
            beam, end_step = completed
            result = self._build_response(beam, end_step)
            result.append(self._eos_token_id)
            results.append(result)

        # Build remaining necessary that are in flight
        more = self._hypothesis - len(results)
        for i in np.argsort(self._scores)[-more:]:
            result = self._build_response(i, len(self._selected_beams))
            results.append(result)

        return results


class LlmDecoder:
    def __init__(
        self,
        decode_config: DecodeConfig,
        prefill_batcher,
        decode_batcher,
        results_callback: Callable[[Union[int, List[int]]], None],
        rid,
        use_native_impls: bool = False,
    ):
        self._decode_config = decode_config
        self._cpp_decode_config = _convert_to_cpp_decode_config(decode_config)
        self._eos_token = self._decode_config.eos_token_id
        self._prefill_batcher = prefill_batcher
        self._decode_batcher = decode_batcher
        self._page_cache = self._decode_batcher.page_cache
        self._tokens_per_page = self._page_cache.tokens_per_page
        self._page_pool = self._page_cache.page_pool
        self._results_callback = results_callback
        self._rid = rid
        self._lock = threading.Lock()
        self._cancelled = False

        if use_native_impls:
            self._select_function = self._native_select
        else:
            self._select_function = (
                select_topk if self._decode_config.num_beams > 1 else select_greedy
            )

        self._score_function = _score_functions[
            self._decode_config.logits_normalization
        ]

    def _native_select(self, logits, decode_config):
        tokens, scores = _sfl.llm.select_tokens(
            logits.flatten(), self._cpp_decode_config
        )
        return np.array(tokens), np.array(scores)

    def cancel(self):
        """Cancel inproceess work."""
        with self._lock:
            self._cancelled = True

    def release(self):
        """Release any remain resources held by the decoder"""
        pass

    def setup_req(self, decode_reqs, tokens, position, page_ids):
        next_token_ids = []

        # TODO: Allocation more requests
        if len(decode_reqs) < len(tokens):
            raise ValueError("NEED TO ALLOCATE MORE REQS")

        for token in tokens:
            next_tokens = [token]
            next_token_ids.append(next_tokens)

        for i, ids in enumerate(next_token_ids):
            decode_reqs[i].input_token_ids = ids
            decode_reqs[i].start_position = position
            decode_reqs[i].page_ids = page_ids[i]

        return decode_reqs[: len(tokens)]

    def create_decode_reqs(self, prefill_req: LlmInferenceExecRequest):
        num_beams = self._decode_config.num_beams
        decode_reqs = [
            LlmInferenceExecRequest(
                input_token_ids=[],
                phase=InferencePhase.DECODE,
                rid=self._rid,
                orig_instance_id=prefill_req.orig_instance_id,
                page_ids=[],
                page_cache=self._page_cache,
            )
            for _ in range(num_beams)
        ]

        for req in decode_reqs:
            req.start_position = len(prefill_req.input_token_ids)

        return decode_reqs

    async def run(self, input_ids):
        input_length = len(input_ids)
        prefill_req = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=input_ids,
            rid=self._rid,
            page_cache=self._prefill_batcher.page_cache,
        )
        prefill_req.acquire_pages()
        # Run Prefill:
        self._prefill_batcher.submit(prefill_req)
        await prefill_req.done

        token_selector = TokenSelector(self._decode_config)
        initial_pages = [p.index for p in prefill_req.allocated_cache_info.pages]
        initial_length = len(prefill_req.input_token_ids)
        page_manager = PageManager(
            self._page_pool,
            initial_pages=initial_pages,
            initial_length=initial_length,
            tokens_per_page=self._tokens_per_page,
        )

        # Run token selection and send to emitter:
        beams, tokens = token_selector.step(
            [prefill_req.result_logits], [prefill_req.result_indices]
        )

        # Setup decode requests:
        decode_reqs = self.create_decode_reqs(prefill_req)

        # Run Decoder:
        for _ in range(self._decode_config.max_completion_tokens - 1):
            if token_selector.done() or self._cancelled or len(beams) == 0:
                break

            # Update the reqs:
            page_ids = page_manager.step_pages(beams)
            to_run = self.setup_req(decode_reqs, tokens, input_length, page_ids)

            input_length = input_length + 1

            self._decode_batcher.reserve_workload(
                rid=prefill_req.orig_instance_id, count=len(to_run)
            )

            for req in to_run:
                req.reset(InferencePhase.DECODE)
                req.update_cache_info()
                self._decode_batcher.submit(req)

            gathered = asyncio.gather(*[req.done for req in to_run])
            await gathered

            # Publish allocated pages for each decode request
            for r in to_run:
                total_tokens = r.start_position + len(r.input_token_ids)
                number_of_complete_pages = (
                    total_tokens // self._decode_batcher.page_seq_stride
                )
                r.publish_allocated_pages(number_of_complete_pages)

            beams, tokens = token_selector.step(
                [req.result_logits for req in to_run],
                [req.result_indices for req in to_run],
            )

        # Remove the reservation:
        self._decode_batcher.reserve_workload(rid=prefill_req.orig_instance_id, count=0)

        # Grab responses:
        completed = token_selector.results()

        # Return Results:
        self._results_callback(completed)

        prefill_req.free_cache_pages()
        page_manager.release_pages()
