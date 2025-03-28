# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import pytest
import random
import struct

from typing import List
from unittest.mock import patch

import shortfin as sf
import shortfin.array as sfnp

from shortfin_apps.utils import convert_int_to_float
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.token_selection_strategy import (
    build_token_selector_config,
    BeamSearchTokenSelectionStrategy,
    DecodeConfig,
    TokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from shortfin_apps.llm.components.token_selection_strategy.beam_group import (
    BeamGroup,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def exec_req_list(exec_req, cache_ref_count, dummy_pages, request):
    num_reqs = request.param if hasattr(request, "param") else len(dummy_pages)
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    exec_reqs = [exec_req]
    with patch.object(exec_req._cache, "fork_pages", return_value=allocation):
        for _ in range(num_reqs - 1):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

    yield exec_reqs


@pytest.fixture(scope="function")
def beam_search_token_selection_strategy():
    yield BeamSearchTokenSelectionStrategy(
        None,
    )


def float_to_float16_int(value):
    # Pack the float into 4 bytes using IEEE 754 single-precision format
    packed = struct.pack(">f", value)
    # Unpack as a 32-bit integer
    i32 = struct.unpack(">I", packed)[0]

    # Extract sign, exponent, and mantissa
    sign = (i32 >> 31) & 0x1
    exponent = (i32 >> 23) & 0xFF
    mantissa = i32 & 0x7FFFFF

    # Adjust exponent for float16 bias (15) and float32 bias (127)
    exponent -= 127 - 15

    if exponent <= 0:
        # Handle subnormal numbers and zero
        f16 = 0
    elif exponent >= 31:
        # Handle infinity and NaN
        f16 = (sign << 15) | (0x1F << 10) | (mantissa >> 13)
    else:
        # Normalized number
        f16 = (sign << 15) | (exponent << 10) | (mantissa >> 13)

    return f16


def test__top_k(device, beam_search_token_selection_strategy):
    # Sorted ascending
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    k = 8
    top_tokens, top_values = beam_search_token_selection_strategy._top_k(src, -k)
    assert top_tokens == [i for i in range(8, 16)]
    assert top_values == [i for i in range(8, 16)]

    # Sorted descending
    data = data[::-1]
    src.items = data
    k = 8
    top_tokens, top_values = beam_search_token_selection_strategy._top_k(src, -k)
    assert sorted(top_tokens) == [i for i in range(0, 8)]
    assert sorted(top_values) == [i for i in range(8, 16)]

    # Randomized data
    random.shuffle(data)
    src.items = data
    k = 5
    expected_values = {val for val in range(11, 16)}
    expected_tokens = [i for i in range(len(data)) if data[i] in expected_values]
    top_tokens, top_values = beam_search_token_selection_strategy._top_k(src, -k)
    assert sorted(top_tokens) == expected_tokens
    assert sorted(top_values) == list(expected_values)


def test__top_k_float16(device, beam_search_token_selection_strategy):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float16)
    data = [float_to_float16_int(float(i)) for i in range(math.prod(src.shape))]
    src.items = data
    k = 8
    top_tokens, top_values = beam_search_token_selection_strategy._top_k(src, -k)
    assert top_tokens == [i for i in range(8, 16)]
    assert top_values == [i for i in range(8, 16)]

    # Randomize data
    random.shuffle(data)
    src.items = data
    k = 5
    expected_values = {val for val in range(11, 16)}
    expected_tokens = [
        i
        for i in range(len(data))
        if convert_int_to_float(data[i], sfnp.float16) in expected_values
    ]
    top_tokens, top_values = beam_search_token_selection_strategy._top_k(src, -k)
    assert sorted(top_tokens) == expected_tokens
    assert sorted(top_values) == list(expected_values)


def test__normalize_exec_req(beam_search_token_selection_strategy, exec_req):
    beam_search_token_selection_strategy.min_log_prob = 42.0

    result = beam_search_token_selection_strategy._normalize_exec_req(exec_req)
    assert result.accumulated_normalization == 42.0


@patch("shortfin.VoidFuture")
def test__final_score(mock_void_future, beam_search_token_selection_strategy):
    initial_prompt = [i for i in range(0, 5)]
    new_input_tokens = [i for i in range(5, 10)]
    score = random.uniform(0, 10)
    accumulated_normalization = random.uniform(10, 20)

    exec_req = LlmInferenceExecRequest(
        InferencePhase.DECODE,
        initial_prompt,
    )
    exec_req.input_token_ids.extend(new_input_tokens)
    exec_req.score = score
    exec_req.accumulated_normalization = accumulated_normalization

    expected = (score - accumulated_normalization) / 5
    final_score = beam_search_token_selection_strategy._final_score(exec_req)
    assert final_score == expected


def test__find_top_beam_completed_reqs(
    beam_search_token_selection_strategy, exec_req_list
):
    scores = [float(val) for val in range(len(exec_req_list))]
    for i, exec_req in enumerate(exec_req_list):
        exec_req.score = scores[i]

    expected_top_beam = exec_req_list[-1]

    # Completed Reqs
    with patch.object(
        beam_search_token_selection_strategy,
        "_final_score",
        side_effect=lambda req: req.score,
    ):
        # Sorted ascending
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            [],
            set(exec_req_list),
        )
        assert top_beam == expected_top_beam

        # Sorted descending
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            [],
            set(exec_req_list[::-1]),
        )
        assert top_beam == expected_top_beam

        # Randomized
        random.shuffle(exec_req_list)
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            [],
            set(exec_req_list),
        )
        assert top_beam == expected_top_beam


def test__find_top_beam_active_reqs(
    beam_search_token_selection_strategy, exec_req_list
):
    scores = [float(val) for val in range(len(exec_req_list))]
    for i, exec_req in enumerate(exec_req_list):
        exec_req.score = scores[i]

    expected_top_beam = exec_req_list[-1]

    # Completed Reqs
    with patch.object(
        beam_search_token_selection_strategy,
        "_final_score",
        side_effect=lambda req: req.score,
    ):
        # Sorted ascending
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            exec_req_list,
            set(),
        )
        assert top_beam == expected_top_beam

        # Sorted descending
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            exec_req_list[::-1],
            set(),
        )
        assert top_beam == expected_top_beam

        # Randomized
        random.shuffle(exec_req_list)
        top_beam = beam_search_token_selection_strategy._find_top_beam(
            exec_req_list,
            set(),
        )
        assert top_beam == expected_top_beam


def test_get_results(beam_search_token_selection_strategy, exec_req_list):
    # Offset the input_ids to differentiate between reqs
    offset = 1
    for exec_req in exec_req_list[1:]:
        exec_req.input_token_ids = [
            token + offset for token in exec_req.input_token_ids
        ]
        offset += 1

    # Add a couple tokens, so that `input_token_ids` > `prompt_length`
    for exec_req in exec_req_list:
        lower_range = exec_req.input_token_ids[-1] + 1
        upper_range = lower_range + 5
        for i in range(lower_range, upper_range):
            exec_req.input_token_ids.append(i)

    num_beams = len(exec_req_list)
    config = TokenSelectionStrategyConfig(
        decode_config=DecodeConfig(
            num_beams=num_beams,
            token_selection_strategy=TokenSelectionStrategy.BEAM_SEARCH,
        ),
        prefill_callback=lambda _: None,
        decode_callback=lambda _: None,
        results_callback=lambda _: None,
        eos_token_id=-1,
        max_completion_tokens=1,
    )
    beam_search_token_selection_strategy._token_selection_strategy_config = config

    expected_results = [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]]

    results = []

    def _results_callback(tokens: List[List[int]]):
        results.extend(tokens)

    beam_search_token_selection_strategy.token_selection_strategy_config.results_callback = (
        _results_callback
    )

    # All completed
    beam_group = BeamGroup(
        eos_token_id=-1,
        num_beams=len(exec_req_list),
        exec_reqs=[],
        selection_callback=lambda _: None,
    )
    beam_group.completed_reqs = set(exec_req_list)

    beam_search_token_selection_strategy.get_results(beam_group)
    assert sorted(results) == expected_results

    # All active
    results = []
    beam_group.completed_reqs = set()
    beam_group.active_exec_reqs = exec_req_list
    beam_search_token_selection_strategy.get_results(beam_group)
    assert sorted(results) == expected_results

    # Mixed
    results = []
    beam_group.completed_reqs = set(exec_req_list[:2])
    beam_group.active_exec_reqs = exec_req_list[2:]
    beam_search_token_selection_strategy.get_results(beam_group)
    assert sorted(results) == expected_results


@pytest.mark.parametrize("exec_req_list", [10], indirect=True)
def test_get_results_extra_reqs(beam_search_token_selection_strategy, exec_req_list):
    # Offset the input_ids to differentiate between reqs
    offset = 1
    for exec_req in exec_req_list[1:]:
        exec_req.input_token_ids = [
            token + offset for token in exec_req.input_token_ids
        ]
        offset += 1

    # Add a couple tokens, so that `input_token_ids` > `prompt_length`
    for exec_req in exec_req_list:
        lower_range = exec_req.input_token_ids[-1] + 1
        upper_range = lower_range + 5
        for i in range(lower_range, upper_range):
            exec_req.input_token_ids.append(i)

    num_beams = 4
    config = TokenSelectionStrategyConfig(
        decode_config=DecodeConfig(
            num_beams=num_beams,
            token_selection_strategy=TokenSelectionStrategy.BEAM_SEARCH,
        ),
        prefill_callback=lambda _: None,
        decode_callback=lambda _: None,
        results_callback=lambda _: None,
        eos_token_id=-1,
        max_completion_tokens=1,
    )
    beam_search_token_selection_strategy._token_selection_strategy_config = config

    expected_results = [
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [9, 10, 11, 12, 13],
    ]

    results = []

    def _results_callback(tokens: List[List[int]]):
        results.extend(tokens)

    beam_search_token_selection_strategy.token_selection_strategy_config.results_callback = (
        _results_callback
    )

    # Completed == `num_beams`
    beam_group = BeamGroup(
        eos_token_id=-1,
        num_beams=num_beams,
        exec_reqs=[],
        selection_callback=lambda _: None,
    )
    beam_group.completed_reqs = set(exec_req_list[:num_beams])
    beam_group.active_exec_reqs = exec_req_list[num_beams:]

    beam_search_token_selection_strategy.get_results(beam_group)
    assert sorted(results) == expected_results

    # Completed < `num_beams`
    results = []
    beam_group.completed_reqs = set(exec_req_list[: num_beams // 2])
    active_reqs = exec_req_list[num_beams // 2 :]
    score = len(active_reqs)
    for req in active_reqs:
        req.score = score
        score -= 1

    beam_group.active_exec_reqs = exec_req_list[num_beams // 2 :]

    beam_search_token_selection_strategy.get_results(beam_group)
    assert len(results) == num_beams
    assert sorted(results) == expected_results


@pytest.mark.asyncio
async def test_beam_search_decode_single(
    cache_ref_count,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    beam_search_token_selection_strategy,
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

    num_beams = 8
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.BEAM_SEARCH,
        num_beams=num_beams,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback,
        decode_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=-1,
        max_completion_tokens=1,
    )

    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        beam_search_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            await beam_search_token_selection_strategy.decode(exec_req)
            assert len(results_array) == num_beams
            logger.info(f"Results: {results_array}")
            expected_value = 15
            for result in results_array:
                assert len(result) == 1
                assert result[0] == expected_value
                expected_value -= 1.0

            fork_pages_mock.call_count == num_beams - 1


@pytest.mark.asyncio
async def test_beam_search_decode_multiple_completions(
    cache_ref_count,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    beam_search_token_selection_strategy,
):
    results_array = []

    def _results_callback(tokens: List[List[int]]):
        results_array.extend(tokens)

    num_beams = 3
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
        nonlocal num_beams
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        for i in range(num_beams):
            data[i] = 42.0

        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.BEAM_SEARCH,
        num_beams=num_beams,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback_multiple_completions,
        decode_callback=_batcher_callback_multiple_completions,
        results_callback=_results_callback,
        eos_token_id=-1,
        max_completion_tokens=5,
    )
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        beam_search_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            await beam_search_token_selection_strategy.decode(exec_req)
            assert len(results_array) == num_beams
            expected_tokens = set([0, 1, 2])
            expected_tail = 0
            results_array = sorted(results_array)
            for result in results_array:
                assert len(result) == config.max_completion_tokens
                assert all(val in expected_tokens for val in result)
                assert result[-1] == expected_tail
                expected_tail += 1

            fork_pages_mock.call_count == num_beams - 1


@pytest.mark.asyncio
async def test_beam_search_decode_eos_token(
    cache_ref_count,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
    beam_search_token_selection_strategy,
):
    results_array = []

    def _results_callback(tokens: List[List[int]]):
        results_array.extend(tokens)

    num_beams = 3
    count = -1

    def _batcher_callback_multiple_completions(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        This functions specifically "rigs" the requests to output an eos
        token at the 5th decode step.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        nonlocal count
        nonlocal num_beams
        nonlocal config
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]

        for i in range(num_beams):
            data[i] = 42.0

        if (count // num_beams) == 3:
            data[num_beams] = 84.0

        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.BEAM_SEARCH,
        num_beams=num_beams,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback_multiple_completions,
        decode_callback=_batcher_callback_multiple_completions,
        results_callback=_results_callback,
        eos_token_id=3,
        max_completion_tokens=10,
    )
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        beam_search_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            await beam_search_token_selection_strategy.decode(exec_req)
            assert len(results_array) == num_beams
            expected_tokens = set([0, 1, 2])
            expected_tail = 3
            results_array = sorted(results_array)
            assert len(results_array) == num_beams
            for result in results_array:
                assert len(result) == 5
                assert all(val in expected_tokens for val in result[:-1])
                assert result[-1] == expected_tail

            fork_pages_mock.call_count == num_beams - 1
