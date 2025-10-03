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
from typing import Dict, List, Tuple
from ..prefill_config import PrefillConfig

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
from shortfin_apps.llm.components.kvcache.attention_cache_abstract import CacheInfo
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    CacheAllocationFailure,
    BasePagedAttentionCache,
)
from shortfin_apps.llm.components.batching.facade import BatchingFacade

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
        page_cache: BasePagedAttentionCache,
        page_pool: PagePool,
        initial_pages: List[int],
        initial_length: int,
        tokens_per_page: int,
    ):
        self._page_cache = page_cache
        self._page_pool = page_pool

        self._free_pages = []
        self._beam_page_ids = [[]]

        self._tokens_per_page = tokens_per_page
        self._allocation_block_size = 8

        self._shared_pages = initial_pages
        self._position = initial_length

        if self._position % self._tokens_per_page > 0:
            self._beam_page_ids[0].append(self._shared_pages[-1])
            self._shared_pages.pop()

    def allocate(
        self,
        req: LlmInferenceExecRequest,
        allocated_cache_recs: Dict[str, CacheInfo],
        input_token_ids: List[int],
        count: int,
        allocate_block: bool = True,
    ):
        req_allocated_cache_info = allocated_cache_recs.get(req.instance_id, None)
        if not req_allocated_cache_info:
            raise CacheAllocationFailure("No allocated cache info found for request.")

        if count > len(self._free_pages):
            acquire_count = max(count, self._allocation_block_size)
            if not allocate_block:
                acquire_count = count

            # do not lookup published tokens as the major performance improvement comes from re-using partially filled pages in prefill phase
            acquired_cache_info = self._page_cache.allocate(
                input_token_ids,
                req_allocated_cache_info,
                acquire_count,
            )

            acquired = acquired_cache_info.pages[len(req_allocated_cache_info.pages) :]
            self._free_pages.extend([p.index for p in acquired])
            pages = req_allocated_cache_info.pages + acquired[:count]
            req_allocated_cache_info = acquired_cache_info
            req_allocated_cache_info.pages = pages
        else:
            req_allocated_cache_info.num_tokens += len(input_token_ids)
            req_allocated_cache_info.tokens.extend(input_token_ids)
            free_pages = self._page_cache.get_allocated_pages(self._free_pages[:count])
            req_allocated_cache_info.pages.extend(free_pages)
        allocation = self._free_pages[:count]
        self._free_pages = self._free_pages[count:]

        return allocation, req

    def _update_decode_reqs_new_page(
        self,
        beam_page_ids: List[List[int]],
        next_token_ids: List[List[int]],
        decode_reqs: List[LlmInferenceExecRequest],
        allocated_cache_recs: Dict[str, CacheInfo],
    ):
        for i, beam in enumerate(beam_page_ids):
            # only do block allocation for the last beam
            pages = []
            if i != len(next_token_ids) - 1:
                pages, req = self.allocate(
                    req=decode_reqs[i],
                    allocated_cache_recs=allocated_cache_recs,
                    input_token_ids=next_token_ids[i],
                    count=1,
                    allocate_block=False,
                )
            else:
                pages, req = self.allocate(
                    req=decode_reqs[i],
                    allocated_cache_recs=allocated_cache_recs,
                    input_token_ids=next_token_ids[i],
                    count=1,
                    allocate_block=True,
                )
            allocated_cache_recs[decode_reqs[i].instance_id] = allocated_cache_recs.get(
                req.instance_id, None
            )
            beam.append(pages[0])

    def _update_decode_reqs_existing_page(
        self,
        beam_page_ids: List[List[int]],
        next_token_ids: List[List[int]],
        decode_reqs: List[LlmInferenceExecRequest],
        allocated_cache_recs: Dict[str, CacheInfo],
    ):
        used = set()
        for i, beam in enumerate(beam_page_ids):
            if len(beam) > 0:
                if beam[-1] in used:
                    new_pages, req = self.allocate(
                        req=decode_reqs[i],
                        allocated_cache_recs=allocated_cache_recs,
                        input_token_ids=next_token_ids[i],
                        count=1,
                        allocate_block=False,
                    )
                    new_page = new_pages[0]

                    allocated_cache_recs[
                        decode_reqs[i].instance_id
                    ] = allocated_cache_recs.get(req.instance_id, None)
                    if beam[-1] != new_page:
                        self._page_pool.copy_page_index(beam[-1], new_page)
                        beam[-1] = new_page
                else:
                    decode_allocated_cache_info = allocated_cache_recs.get(
                        decode_reqs[i].instance_id, None
                    )
                    if not decode_allocated_cache_info:
                        raise CacheAllocationFailure(
                            "No allocated cache info found for request."
                        )

                    decode_allocated_cache_info.num_tokens += len(next_token_ids[i])
                    decode_allocated_cache_info.tokens.extend(next_token_ids[i])

                used.add(beam[-1])

    def update_decode_reqs(
        self,
        select: List[int],
        decode_reqs: List[LlmInferenceExecRequest],
        allocated_cache_recs: Dict[str, CacheInfo],
        tokens: List[int],
        position: int,
    ) -> List[LlmInferenceExecRequest]:
        # TODO: Allocation more requests
        if len(decode_reqs) < len(tokens):
            raise ValueError("NEED TO ALLOCATE MORE REQS")

        next_token_ids = []
        for token in tokens:
            next_tokens = [token]
            next_token_ids.append(next_tokens)
        if len(select) == 0:
            return
        new_page = (self._position % self._tokens_per_page) == 0
        new_beam_page_ids = [[p for p in self._beam_page_ids[b]] for b in select]
        old_pages = set(itertools.chain.from_iterable(self._beam_page_ids))
        new_pages = set(itertools.chain.from_iterable(new_beam_page_ids))
        free_pages = old_pages - new_pages
        self._free_pages.extend(free_pages)

        if new_page:
            self._update_decode_reqs_new_page(
                new_beam_page_ids, next_token_ids, decode_reqs, allocated_cache_recs
            )
        else:
            self._update_decode_reqs_existing_page(
                new_beam_page_ids, next_token_ids, decode_reqs, allocated_cache_recs
            )

        self._beam_page_ids = new_beam_page_ids
        self._position += 1

        # setup decode_reqs
        for i, ids in enumerate(next_token_ids):
            decode_reqs[i].input_token_ids = ids
            decode_reqs[i].start_position = position
            decode_reqs[i].page_ids = self._shared_pages + new_beam_page_ids[i]
        return decode_reqs[: len(tokens)]

    def release_pages(self):
        self._page_cache.free_allocated_pages(self._free_pages)
        self._free_pages = []


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
        for i, completed in enumerate(self._completed):
            beam, end_step = completed
            result = self._build_response(beam, end_step)
            result.append(self._eos_token_id)
            results.append(result)
            if i == self._hypothesis - 1:
                break

        # Build remaining necessary that are in flight
        if len(results) < self._hypothesis:
            more = self._hypothesis - len(results)
            for i in np.argsort(self._scores)[-more:]:
                result = self._build_response(i, len(self._selected_beams))
                results.append(result)

        return results


class LlmDecoder:
    def __init__(
        self,
        prefill_config: PrefillConfig,
        decode_config: DecodeConfig,
        unified_batcher: BatchingFacade,
        results_callback: Callable[[Union[int, List[int]]], None],
        rid,
        use_native_impls: bool = False,
    ):
        self._prefill_config = prefill_config
        self._decode_config = decode_config
        self._cpp_decode_config = _convert_to_cpp_decode_config(decode_config)
        self._eos_token = self._decode_config.eos_token_id
        self._unified_batcher = unified_batcher
        self._page_cache = self._unified_batcher.get_page_cache()
        self._tokens_per_page = self._page_cache.tokens_per_page
        self._page_pool = self._page_cache.page_pool
        self._results_callback = results_callback
        self._rid = rid
        self._lock = threading.Lock()
        self._cancelled = False
        self._allocated_cach_recs: Dict[str, CacheInfo] = {}

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

    def create_decode_reqs(self, prefill_req: LlmInferenceExecRequest):
        num_beams = self._decode_config.num_beams
        decode_reqs = [
            LlmInferenceExecRequest(
                input_token_ids=[],
                phase=InferencePhase.DECODE,
                rid=self._rid,
                orig_instance_id=prefill_req.orig_instance_id,
                page_ids=[],
            )
            for _ in range(num_beams)
        ]

        for req in decode_reqs:
            req.start_position = len(prefill_req.input_token_ids)
            self._allocated_cach_recs[req.instance_id] = self._allocated_cach_recs[
                prefill_req.instance_id
            ]

        return decode_reqs

    def create_prefill_req(self, input_ids):
        prefill_req = LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL, input_token_ids=input_ids, rid=self._rid
        )

        cached_allocation = self._page_cache.lookup(input_ids)
        token_ids = input_ids[cached_allocation.num_tokens :]
        allocated_cache_info = self._page_cache.allocate(token_ids, cached_allocation)
        prefill_req.page_ids = [p.index for p in allocated_cache_info.pages]

        # TODO(stbaione): Extend for non-zero start positions
        # when `trie` changes are landed.
        if self._prefill_config.has_prefill_position:
            prefill_req.start_position = 0

        # add allocated cache info to the dictionary
        self._allocated_cach_recs[prefill_req.instance_id] = allocated_cache_info

        return prefill_req

    def publish_request(
        self, req: LlmInferenceExecRequest, publish_incomplete_page: bool = False
    ):
        req_cache_info = self._allocated_cach_recs.get(req.instance_id, None)
        if not req_cache_info:
            return

        updated_cache_info = self._page_cache.publish_pages_for_tokens(
            req_cache_info, publish_incomplete_page=publish_incomplete_page
        )
        self._allocated_cach_recs[req.instance_id] = updated_cache_info

    def free_req_cache(self, req: LlmInferenceExecRequest):
        req_cache_info = self._allocated_cach_recs.get(req.instance_id, None)
        if not req_cache_info:
            return

        self._page_cache.release_pages(req_cache_info)
        req.page_ids = []
        self._allocated_cach_recs[req.instance_id] = None

    async def run(self, input_ids):
        input_length = len(input_ids)
        prefill_req = self.create_prefill_req(input_ids)
        # Run Prefill:
        self._unified_batcher.submit(prefill_req)
        await prefill_req.done
        self.publish_request(prefill_req, publish_incomplete_page=False)

        token_selector = TokenSelector(self._decode_config)
        prefill_req_cache_info = self._allocated_cach_recs.get(
            prefill_req.instance_id, None
        )
        if not prefill_req_cache_info:
            raise CacheAllocationFailure(
                "No allocated cache info found for prefill request."
            )

        initial_pages = [p.index for p in prefill_req_cache_info.pages]
        initial_length = len(prefill_req.input_token_ids)
        page_manager = PageManager(
            self._page_cache,
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
            to_run = page_manager.update_decode_reqs(
                beams, decode_reqs, self._allocated_cach_recs, tokens, input_length
            )

            input_length = input_length + 1

            self._unified_batcher.reserve_workload(
                rid=prefill_req.orig_instance_id, count=len(to_run)
            )

            for req in to_run:
                req.reset(InferencePhase.DECODE)
                self._unified_batcher.submit(req)

            gathered = asyncio.gather(*[req.done for req in to_run])
            await gathered

            beams, tokens = token_selector.step(
                [req.result_logits for req in to_run],
                [req.result_indices for req in to_run],
            )

        # Remove the reservation:
        self._unified_batcher.reserve_workload(
            rid=prefill_req.orig_instance_id, count=0
        )

        # Grab responses:
        completed = token_selector.results()

        # Return Results:
        self._results_callback(completed)

        for req in decode_reqs:
            self.publish_request(req, publish_incomplete_page=True)
            self.free_req_cache(req)
        page_manager.release_pages()
