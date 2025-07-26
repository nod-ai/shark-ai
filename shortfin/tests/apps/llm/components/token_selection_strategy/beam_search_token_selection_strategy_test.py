# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import numpy as np
import pytest
import random
import struct

from typing import List
from unittest.mock import patch

import shortfin.array as sfnp

from shortfin_apps.utils import approximately_equal
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components.token_selection_strategy import (
    build_token_selector_config,
    BeamSearchScorer,
    TokenSelector,
    DecodeConfig,
    TokenSelectionStrategyConfig,
)
from shortfin_apps.llm.components.token_selection_strategy.beam_group import (
    BeamGroup,
    BeamSearchBeam,
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
def num_beams():
    return 3


@pytest.fixture(scope="function")
def decode_config(num_beams):
    yield DecodeConfig(
        num_beams=num_beams,
        use_beam_search=True,
        max_completion_tokens=1,
        eos_token_id=-1,
    )


@pytest.fixture(scope="function")
def beam_search_scorer(num_beams: int):
    yield BeamSearchScorer(num_beams=num_beams)


@pytest.fixture(scope="function")
def beam_search_token_selection_strategy():
    yield TokenSelector(
        token_selection_strategy_config=None,
    )


@pytest.fixture(scope="function")
def beam_search_beam(exec_req, decode_config):
    decode_config.use_beam_search = True
    yield BeamSearchBeam(
        exec_req,
        decode_config=decode_config,
    )


class FakeBatcher:
    def __init__(self, submit_cb, workitem_cb):
        self.submit = submit_cb
        self.reserve_workload = workitem_cb


def _batcher_workitem_callback(rid: int, count: int):
    pass


def float_to_float16_int(value: float):
    packed_val = struct.pack("<e", value)
    return struct.unpack("<H", packed_val)[0]


def test_beam_search_beam_sample_logits(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.decode_config.num_beams = 3
    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    expected_tokens = sfnp.argpartition(src, -3, -1).view(0, 0, slice(-3, None))
    assert top_tokens.tolist() == expected_tokens.items.tolist()


def test_beam_search_beam_sample_logits_w_indices(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    random.shuffle(data)
    src.items = data

    indices_np = np.flip(np.argpartition(np.array(src), -3, -1), axis=-1)
    indices = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    indices.items = indices_np.flatten().tolist()

    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.exec_req.result_indices = indices
    beam_search_beam.decode_config.num_beams = 3
    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    expected_tokens = indices.view(0, 0, slice(None, 3))
    assert top_tokens.tolist() == expected_tokens.items.tolist()


def test_beam_search_beam_sample_logits_top_k(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.decode_config.num_beams = 3

    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    assert top_tokens.tolist() == [13, 14, 15]

    beam_search_beam.decode_config.top_k = 3
    expected_tokens = top_tokens.tolist()
    values = [0.33] * 3
    expected_values = [math.log(0.33)] * 3
    with patch.object(
        beam_search_beam, "_sample_logits_top_k", return_value=(expected_tokens, values)
    ):
        result_tokens, result_values = beam_search_beam.sample_logits(3)

        assert result_tokens == expected_tokens
        assert approximately_equal(result_values, expected_values)


def test_beam_search_beam_sample_logits_top_k_w_indices(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    indices_np = np.flip(
        np.argpartition(src, -3, -1),
        axis=-1,
    )
    indices = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    indices.items = indices_np.flatten().tolist()

    beam_search_beam.decode_config.top_k = 3
    beam_search_beam.decode_config.num_beams = 3
    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.exec_req.result_indices = indices

    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    expected_tokens = indices.view(0, 0, slice(None, 3)).items.tolist()
    for token in top_tokens.tolist():
        assert token in expected_tokens


def test_beam_search_beam_sample_logits_top_p(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    beam_search_beam.exec_req.result_logits = src

    top_tokens, _ = beam_search_beam.sample_logits(3)

    beam_search_beam.decode_config.top_p = 0.95
    beam_search_beam.decode_config.top_k = None
    expected_tokens = top_tokens.tolist()
    values = [0.33] * 3
    expected_values = [math.log(0.33)] * 3
    with patch.object(
        beam_search_beam, "_sample_logits_top_p", return_value=(expected_tokens, values)
    ):
        result_tokens, result_values = beam_search_beam.sample_logits(0)

        assert result_tokens == expected_tokens
        assert approximately_equal(result_values, expected_values)


def test_beam_search_beam_sample_logits_top_p_w_indices(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [0] * math.prod(src.shape)
    data[0:3] = [4.41] * 3
    src.items = data

    indices_np = np.flip(
        np.argpartition(src, -3, -1),
        axis=-1,
    )
    indices = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    indices.items = indices_np.flatten().tolist()

    beam_search_beam.decode_config.top_p = 0.94
    beam_search_beam.decode_config.num_beams = 3
    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.exec_req.result_indices = indices

    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    expected_tokens = indices.view(0, 0, slice(None, 3)).items.tolist()
    for token in top_tokens.tolist():
        assert token in expected_tokens


def test_beam_search_beam_sample_logits_top_k_top_p(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    beam_search_beam.exec_req.result_logits = src
    top_tokens, _ = beam_search_beam.sample_logits(3)

    beam_search_beam.decode_config.top_k = 5
    beam_search_beam.decode_config.top_p = 0.95

    top_k_tokens = top_tokens.tolist() + [12, 11]
    top_k_values = ([0.33] * 3) + [0.0, 0.0]
    expected_tokens = top_tokens.tolist()
    values = [0.33] * 3
    expected_values = [math.log(0.33)] * 3

    with patch.object(
        beam_search_beam,
        "_sample_logits_top_k",
        return_value=(top_k_tokens, top_k_values),
    ):
        with patch.object(
            beam_search_beam,
            "_sample_logits_top_p",
            return_value=(expected_tokens, values),
        ):
            result_tokens, result_values = beam_search_beam.sample_logits(3)
            assert result_tokens == expected_tokens
            assert approximately_equal(result_values, expected_values)


def test_beam_search_beam_sample_logits_top_k_top_p_w_indices(device, beam_search_beam):
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    # data = [float(i) for i in range(math.prod(src.shape))]
    data = [0] * math.prod(src.shape)
    data[0:3] = [4.41] * 3
    src.items = data

    indices_np = np.flip(
        np.argpartition(src, -3, -1),
        axis=-1,
    )
    indices = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    indices.items = indices_np.flatten().tolist()

    beam_search_beam.decode_config.top_k = 5
    beam_search_beam.decode_config.top_p = 0.94
    beam_search_beam.decode_config.num_beams = 3
    beam_search_beam.exec_req.result_logits = src
    beam_search_beam.exec_req.result_indices = indices

    top_tokens, top_values = beam_search_beam.sample_logits(0)

    assert len(top_tokens) == 3
    assert len(top_values) == 3

    expected_tokens = indices.view(0, 0, slice(None, 3)).items.tolist()
    for token in top_tokens.tolist():
        assert token in expected_tokens


def test_beam_search_beam_update_score(beam_search_beam, beam_search_scorer):
    score = 42.0
    beam_search_scorer.update_score(beam_search_beam, score)
    assert beam_search_beam.score == 42.0


def test_beam_search_beam_update_exec_req(beam_search_beam):
    expected_start_position = beam_search_beam.exec_req.start_position + 1
    expected_token = 42

    beam_search_beam.last_token = expected_token
    beam_search_beam.update_exec_req()

    assert beam_search_beam.exec_req.input_token_ids[-1] == expected_token
    assert beam_search_beam.exec_req.start_position == expected_start_position


def test_beam_search_beam_normalize_score(beam_search_beam, beam_search_scorer):
    min_log_prob = -42.0
    beam_search_scorer.normalize_score(beam_search_beam, min_log_prob)
    assert beam_search_beam.accumulated_normalization == 42.0


@patch("shortfin.VoidFuture")
def test_beam_search_beam_update_final_score(
    mock_void_future, decode_config, beam_search_scorer
):
    initial_prompt = [i for i in range(0, 5)]
    new_input_tokens = [i for i in range(5, 10)]
    score = random.uniform(0, 10)
    accumulated_normalization = random.uniform(10, 20)

    exec_req = LlmInferenceExecRequest(
        InferencePhase.DECODE,
        initial_prompt,
    )
    exec_req.input_token_ids.extend(new_input_tokens)
    beam = BeamSearchBeam(
        exec_req,
        decode_config=decode_config,
        score=score,
        accumulated_normalization=accumulated_normalization,
    )

    expected = (score - accumulated_normalization) / 5
    beam_search_scorer.finalize_score(beam)
    assert beam.score == expected


def test_get_results(
    beam_search_token_selection_strategy, decode_config, exec_req_list
):
    decode_config.use_beam_search = True
    beams = [
        BeamSearchBeam(exec_req, decode_config=decode_config)
        for exec_req in exec_req_list
    ]
    # Offset the input_ids to differentiate between reqs
    offset = 1
    for beam in beams[1:]:
        beam.exec_req.input_token_ids = [
            token + offset for token in beam.exec_req.input_token_ids
        ]
        offset += 1

    # Add a couple tokens, so that `input_token_ids` > `prompt_length`
    for beam in beams:
        exec_req = beam.exec_req
        lower_range = exec_req.input_token_ids[-1] + 1
        upper_range = lower_range + 5
        for i in range(lower_range, upper_range):
            exec_req.input_token_ids.append(i)

    config = TokenSelectionStrategyConfig(
        decode_config=decode_config,
        prefill_callback=lambda _: None,
        decode_callback=lambda _: None,
        results_callback=lambda _: None,
        decode_reserve_callback=lambda _: None,
    )
    beam_search_token_selection_strategy.token_selection_strategy_config = config

    expected_results = [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]]

    results = []

    def _results_callback(tokens: List[List[int]]):
        results.extend(tokens)

    beam_search_token_selection_strategy.token_selection_strategy_config.results_callback = (
        _results_callback
    )

    # All completed
    beam_group = BeamGroup(
        exec_req_list[0],
        decode_config,
        beams=beams,
    )
    beam_group._completed_beams = beams
    beam_group._active_beams = []

    results = beam_group.get_results()
    assert results == expected_results

    # All active
    beam_group = BeamGroup(
        exec_req_list[0],
        decode_config,
        beams=beams,
    )
    beam_group._completed_beams = []
    beam_group._active_beams = beams
    results = beam_group.get_results()
    assert results == expected_results

    # Mixed
    results = []
    beam_group._completed_beams = beams[:2]
    beam_group._active_beams = beams[2:]
    results = beam_group.get_results()
    assert results == expected_results


@pytest.mark.parametrize("exec_req_list", [10], indirect=True)
def test_get_results_extra_reqs(
    beam_search_token_selection_strategy, decode_config, exec_req_list
):
    beams = [
        BeamSearchBeam(exec_req, decode_config=decode_config)
        for exec_req in exec_req_list
    ]
    # Offset the input_ids to differentiate between reqs
    offset = 1
    for beam in beams[1:]:
        beam.exec_req.input_token_ids = [
            token + offset for token in beam.exec_req.input_token_ids
        ]
        offset += 1

    # Add a couple tokens, so that `input_token_ids` > `prompt_length`
    for beam in beams:
        exec_req = beam.exec_req
        lower_range = exec_req.input_token_ids[-1] + 1
        upper_range = lower_range + 5
        for i in range(lower_range, upper_range):
            exec_req.input_token_ids.append(i)

    num_beams = 3
    config = TokenSelectionStrategyConfig(
        decode_config=decode_config,
        prefill_callback=lambda _: None,
        decode_callback=lambda _: None,
        decode_reserve_callback=lambda _: None,
        results_callback=lambda _: None,
    )
    beam_search_token_selection_strategy.token_selection_strategy_config = config

    expected_results = [
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
    ]

    results = []

    def _results_callback(tokens: List[List[int]]):
        results.extend(tokens)

    beam_search_token_selection_strategy.token_selection_strategy_config.results_callback = (
        _results_callback
    )

    # Completed == `num_beams`
    beam_group = BeamGroup(
        exec_req_list[0],
        decode_config,
        beams=beams,
    )
    beam_group._completed_beams = beams[:num_beams]
    beam_group._active_beams = beams[num_beams:]

    results = beam_group.get_results()
    assert results == expected_results

    # Completed < `num_beams`
    results = []
    beam_group._completed_beams = beams[: num_beams // 2]
    active_beams = beams[num_beams // 2 :]
    score = len(active_beams)
    for beam in active_beams:
        beam.score = score
        score -= 1

    beam_group._active_beams = beams[num_beams // 2 :]

    results = beam_group.get_results()
    assert len(results) == num_beams
    assert results == expected_results


@pytest.mark.asyncio
async def test_beam_search_decode_single(
    cache_ref_count,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
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
        use_beam_search=True,
        num_beams=num_beams,
        max_completion_tokens=2,
        eos_token_id=-1,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        decode_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        results_callback=_results_callback,
    )
    beam_search_token_selection_strategy = TokenSelector(
        token_selection_strategy_config=config,
    )
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        beam_search_token_selection_strategy,
        "token_selection_strategy_config",
        new=config,
    ):
        with patch.object(
            exec_req._cache, "fork_pages", return_value=allocation
        ) as fork_pages_mock:
            with patch.object(
                BeamGroup,
                "clean_up",
            ) as mock_clean_up:
                await beam_search_token_selection_strategy.decode(exec_req)
                assert len(results_array) == num_beams
                expected_value = 15
                for result in results_array:
                    assert len(result) == 1
                    assert result[0] == expected_value
                    expected_value -= 1.0

                fork_pages_mock.call_count == num_beams - 1
                mock_clean_up.assert_called_once()


@pytest.mark.asyncio
async def test_beam_search_decode_multiple_completions(
    cache_ref_count,
    device,
    dummy_pages,
    exec_req: LlmInferenceExecRequest,
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
        use_beam_search=True,
        num_beams=num_beams,
        max_completion_tokens=5,
        eos_token_id=-1,
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
    )
    beam_search_token_selection_strategy = TokenSelector(
        token_selection_strategy_config=config,
    )
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        exec_req._cache, "fork_pages", return_value=allocation
    ) as fork_pages_mock:
        with patch.object(
            BeamGroup,
            "clean_up",
        ) as mock_clean_up:
            await beam_search_token_selection_strategy.prefill(exec_req)
            await beam_search_token_selection_strategy.decode(exec_req)
            assert len(results_array) == num_beams
            expected_tokens = set([0, 1, 2])
            expected_tail = 0
            results_array = sorted(results_array)
            for result in results_array:
                assert len(result) == config.decode_config.max_completion_tokens
                assert all(val in expected_tokens for val in result)
                assert result[-1] == expected_tail
                expected_tail += 1

            fork_pages_mock.call_count == num_beams - 1
            mock_clean_up.assert_called_once()


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
        use_beam_search=True,
        num_beams=num_beams,
        max_completion_tokens=10,
        eos_token_id=3,
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
    )
    beam_search_token_selection_strategy = TokenSelector(
        token_selection_strategy_config=config,
    )
    exec_req._cache = cache_ref_count
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache_ref_count)
    exec_req.allocation = allocation
    with patch.object(
        exec_req._cache, "fork_pages", return_value=allocation
    ) as fork_pages_mock:
        with patch.object(
            BeamGroup,
            "clean_up",
        ) as mock_clean_up:
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
            mock_clean_up.assert_called_once()
