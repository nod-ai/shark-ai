# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import pytest
from typing import List
from unittest.mock import patch

import shortfin as sf
import shortfin.array as sfnp

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.token_selection_strategy.beam_group import (
    BeamGroup,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components.token_selection_strategy import (
    build_token_selector_config,
    DecodeConfig,
    MultiGreedyTokenSelectionStrategy,
    TokenSelectionStrategy,
)
from shortfin_apps.llm.components.token_selection_strategy.greedy_token_selection_strategy import (
    GreedyBeam,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def exec_req_list(exec_req, cache, dummy_pages):
    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    exec_reqs = [exec_req]
    num_beams = len(dummy_pages)
    with patch.object(exec_req._cache, "fork_pages", return_value=allocation):
        for _ in range(num_beams - 1):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

    yield exec_reqs


@pytest.fixture(scope="function")
def multi_greedy_token_selection_strategy():
    yield MultiGreedyTokenSelectionStrategy(
        None,
    )


@pytest.fixture(scope="function")
def greedy_beam(exec_req, decode_config):
    yield GreedyBeam(
        exec_req,
        decode_config=decode_config,
    )


class FakeBatcher:
    def __init__(self, submit_cb, workitem_cb):
        self.submit = submit_cb
        self.reserve_workitem = workitem_cb
        self.complete_workitem = workitem_cb


def _batcher_workitem_callback(_: int):
    pass


def test_select_greedy(
    decode_config, device, exec_req_list, multi_greedy_token_selection_strategy
):
    count = 0
    for exec_req in exec_req_list:
        src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(src.shape))]
        data[count] = 42.0
        src.items = data
        exec_req.result_logits = src
        count += 1

    beams = [
        GreedyBeam(exec_req, decode_config=decode_config) for exec_req in exec_req_list
    ]
    selections = multi_greedy_token_selection_strategy.select_greedy(beams, [])
    assert len(selections) == len(beams)

    expected_last_tokens = [i for i in range(len(beams))]
    assert [selection.last_token for selection in selections] == expected_last_tokens


@pytest.mark.asyncio
async def test_multi_greedy_decode_single(
    cache,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    multi_greedy_token_selection_strategy,
):
    def _batcher_callback(request: LlmInferenceExecRequest):
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()

    results_array = []

    def _results_callback(tokens: List[List[int]]):
        results_array.extend(tokens)

    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.MULTI_GREEDY,
        num_beams=2,
        max_completion_tokens=1,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        decode_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        results_callback=_results_callback,
        eos_token_id=-1,
    )

    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    with patch.object(
        multi_greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            with patch.object(
                BeamGroup,
                "clean_up",
            ) as mock_clean_up:
                await multi_greedy_token_selection_strategy.decode(exec_req)
                logger.info(f"results_array: {results_array}")
                assert len(results_array) == 2
                for result in results_array:
                    assert len(result) == 1
                    assert result[0] == 15

                fork_pages_mock.assert_called_once()
                mock_clean_up.assert_called_once()


@pytest.mark.asyncio
async def test_multi_greedy_decode_multiple_completions(
    cache,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    multi_greedy_token_selection_strategy,
):
    results_array = []

    def _results_callback(tokens: List[List[int]]):
        results_array.extend(tokens)

    count = 0

    def _batcher_callback_multiple_completions(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        nonlocal count
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        # Set max to an explicit index
        data[count // 2] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.MULTI_GREEDY,
        num_beams=2,
        max_completion_tokens=5,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, _batcher_workitem_callback
        ),
        decode_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, _batcher_workitem_callback
        ),
        results_callback=_results_callback,
        eos_token_id=-1,
    )

    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    with patch.object(
        multi_greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            with patch.object(
                BeamGroup,
                "clean_up",
            ) as mock_clean_up:
                await multi_greedy_token_selection_strategy.decode(exec_req)
                assert len(results_array) == 2
                for result in results_array:
                    assert len(result) == 5
                    assert result == [0, 1, 2, 3, 4]

                fork_pages_mock.assert_called_once()
                mock_clean_up.assert_called_once()


@pytest.mark.asyncio
async def test_multi_greedy_decode_eos_token(
    cache,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    multi_greedy_token_selection_strategy,
):
    results_array = []

    def _results_callback(tokens: List[int]):
        results_array.extend(tokens)

    count = 0

    def _batcher_callback_multiple_completions(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        nonlocal count
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        # Set max to an explicit index
        data[count // 2] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.MULTI_GREEDY,
        num_beams=2,
        max_completion_tokens=5,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, _batcher_workitem_callback
        ),
        decode_batcher=FakeBatcher(
            _batcher_callback_multiple_completions, _batcher_workitem_callback
        ),
        results_callback=_results_callback,
        eos_token_id=-1,
    )

    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    with patch.object(
        multi_greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            with patch.object(
                BeamGroup,
                "clean_up",
            ) as mock_clean_up:
                await multi_greedy_token_selection_strategy.decode(exec_req)
                logger.info(f"results_array: {results_array}")
                assert len(results_array) == 2
                for result in results_array:
                    assert len(result) == 5
                    assert result == [0, 1, 2, 3, 4]

                fork_pages_mock.assert_called_once()
                mock_clean_up.assert_called_once()
