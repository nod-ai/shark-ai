# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import numpy as np
import operator
import threading

from copy import deepcopy
from functools import reduce
from shortfin_apps.llm.components.kvcache.page_pool import PagePool
from shortfin_apps.llm.components.token_selection_strategy.config import (
    DecodeConfig,
    LogitsNormalization,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from typing import Callable, List, Union


def combine_scores_null(old_score: np.ndarray, step: np.ndarray, norm: float):
    new_score = old_score + step
    new_score = new_score - norm
    return new_score


def combine_scores_softmax(old_score: np.ndarray, step: np.ndarray, norm: float):
    new_score = old_score * step
    new_score = new_score / max(norm, 0.1)
    return new_score


def combine_scores_log_softmax(old_score: np.ndarray, step: np.ndarray, norm: float):
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
    argmax = argmax[None]

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
    def __init__(self, page_pool: PagePool):
        self._page_pool = page_pool
        self._allocated_pages = []
        self._allocated_page_ids = set()

        self._block_allocation = 8

    def allocate_more(self, count):
        count = max(self._block_allocation, count)
        new_pages = self._page_pool.acquire_free_pages(count)
        new_page_ids = [p.index for p in new_pages]
        self._allocated_pages.extend(new_pages)
        self._allocated_page_ids.update(new_page_ids)

        return new_page_ids

    def reallocate_pages(
        self,
        new_page_sets: list[list[int]],
        add_empty_pages: bool = False,
    ) -> list[list[int]]:
        used_pages = reduce(operator.__or__, [set(l) for l in new_page_sets])
        new_page_sets = [deepcopy(p) for p in new_page_sets]
        free_pages = self._allocated_page_ids - used_pages

        allocate_pages = len(free_pages) < len(new_page_sets)
        if allocate_pages > 0:
            new_page_ids = self.allocate_more(allocate_pages)
            free_pages.update(new_page_ids)

        # We just need to add a new page:
        if add_empty_pages:
            for page_set in new_page_sets:
                page_set.append(free_pages.pop())

        # Copy the final pages if possibly duplicate:
        if not add_empty_pages:
            used_final_pages = set()
            for page_set in new_page_sets:
                last_page = page_set[-1]
                if last_page in used_final_pages:
                    new_page = free_pages.pop()
                    page_set[-1] = new_page
                    self._page_pool.copy_page_index(
                        src_page=last_page, dst_page=new_page
                    )
                used_final_pages.add(last_page)

        return new_page_sets

    def release_pages(self):
        self._page_pool.free_pages(self._allocated_pages)
        self._allocated_pages = []


class LlmDecoder:
    def __init__(
        self,
        decode_config: DecodeConfig,
        prefill_batcher,
        decode_batcher,
        results_callback: Callable[[Union[int, List[int]]], None],
        rid,
    ):
        self._decode_config = decode_config
        self._eos_token = self._decode_config.eos_token_id
        self._prefill_batcher = prefill_batcher
        self._decode_batcher = decode_batcher
        self._page_cache = self._decode_batcher.page_cache
        self._tokens_per_page = self._page_cache.tokens_per_page
        self._page_pool = self._page_cache.page_pool
        self._results_callback = results_callback
        self._rid = rid
        self._page_manager = PageManager(self._page_pool)
        self._lock = threading.Lock()
        self._cancelled = False

        self._select_function = (
            select_topk if self._decode_config.use_beam_search else select_greedy
        )

        self._score_function = _score_functions[
            self._decode_config.logits_normalization
        ]

    def cancel(self):
        """Cancel inproceess work."""
        with self._lock:
            self._cancelled = True

    def release(self):
        """Release any remain resources held by the decoder"""
        self._page_manager.release_pages()

    def select(self, reqs):
        # Setup next steps:
        max_score = max(req.score for req in reqs)
        logits = [
            self._score_function(np.asarray(req.result_logits), req.score, max_score)
            for req in reqs
        ]
        logits = np.concatenate(logits, axis=1)[0]

        token_options = logits.shape[-1]
        tokens, scores = self._select_function(logits, self._decode_config)

        indices = [np.asarray(req.result_indices) for req in reqs]

        if indices[0] is not None:
            indices = np.concatenate(indices, axis=1)[0]
            beams = tokens // token_options
            tokens = np.take(indices, tokens)
        else:
            beams = tokens // token_options
            tokens = tokens % token_options

        return beams, tokens, scores

    def setup_req(self, decode_reqs, beams, tokens, scores):
        next_token_ids = []
        next_page_ids = []

        # TODO: Allocation more requests
        if len(decode_reqs) < tokens.shape[0]:
            raise ValueError("NEED TO ALLOCATE MORE REQS")

        completed = []
        for beam, token in zip(beams, tokens):
            req = decode_reqs[beam]
            next_tokens = req.input_token_ids + [token]
            if token == self._eos_token:
                completed.append(next_tokens)
                continue

            next_page_ids.append(req.page_ids)
            next_token_ids.append(next_tokens)

        required_blocks = -(len(next_token_ids[0]) // -self._tokens_per_page)
        add_empty_pages = required_blocks > len(next_page_ids[0])

        next_page_ids = self._page_manager.reallocate_pages(
            new_page_sets=next_page_ids,
            add_empty_pages=add_empty_pages,
        )

        for i, ids in enumerate(next_token_ids):
            decode_reqs[i].input_token_ids = ids
            decode_reqs[i].start_position = len(ids) - 1
            decode_reqs[i].page_ids = next_page_ids[i]
            decode_reqs[i].score = scores[i]

        return decode_reqs[: tokens.shape[0]], completed

    def create_decode_reqs(self, prefill_req: LlmInferenceExecRequest):
        num_beams = (
            self._decode_config.num_beams if self._decode_config.use_beam_search else 1
        )
        input_token_ids = deepcopy(prefill_req.input_token_ids)
        page_ids = [p.index for p in prefill_req.allocation.pages]
        decode_reqs = [
            LlmInferenceExecRequest(
                input_token_ids=input_token_ids,
                phase=InferencePhase.DECODE,
                rid=self._rid,
                orig_instance_id=prefill_req.orig_instance_id,
                page_ids=page_ids,
            )
            for _ in range(num_beams)
        ]

        return decode_reqs

    async def run(self, input_ids):
        prefill_req = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=input_ids,
            rid=self._rid,
        )
        prompt_length = len(input_ids)
        prefill_req._cache = self._page_cache

        # Run Prefill:
        self._prefill_batcher.submit(prefill_req)
        await prefill_req.done

        # Run token selection and send to emitter:
        beams, tokens, scores = self.select([prefill_req])

        # Decode requests:
        decode_reqs = self.create_decode_reqs(prefill_req)

        # Update the reqs:
        to_run, completed = self.setup_req(decode_reqs, beams, tokens, scores)

        # Run Decoder:
        for _ in range(self._decode_config.max_completion_tokens - 1):
            self._decode_batcher.reserve_workload(
                rid=prefill_req.orig_instance_id, count=len(to_run)
            )

            for req in to_run:
                req.reset(InferencePhase.DECODE)
                self._decode_batcher.submit(req)

            gathered = asyncio.gather(*[req.done for req in to_run])
            await gathered

            beams, tokens, scores = self.select(to_run)

            to_run, new_completed = self.setup_req(decode_reqs, beams, tokens, scores)
            completed.extend(new_completed)

            if len(completed) > self._decode_config.num_beams:
                break

            if self._cancelled:
                break

        # Remove the reservation:
        self._decode_batcher.reserve_workload(rid=prefill_req.orig_instance_id, count=0)

        # Finish out with uncomplete responses:
        incomplete_needed = max(self._decode_config.num_beams - len(completed), 0)
        incomplete_responses = [
            req.input_token_ids for req in to_run[:incomplete_needed]
        ]
        completed.extend(incomplete_responses)

        completed = [resp[prompt_length:] for resp in completed]

        # Return Results:
        self._results_callback(completed)

        prefill_req.free_cache_pages()
        self._page_manager.release_pages()
