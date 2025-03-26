# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import pytest
from unittest.mock import patch

import shortfin.array as sfnp

from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components.token_selection_strategy import (
    build_token_selector_config,
    GreedyTokenSelectionStrategy,
    TokenSelectionStrategy,
    DecodeConfig,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def greedy_token_selection_strategy():
    yield GreedyTokenSelectionStrategy(
        None,
    )


@pytest.mark.asyncio
async def test_greedy_decode_single(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    def _batcher_callback(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `TokenSelectionStrategy.prefill`.

        This adds a `device_array` to the `LlmInferenceExecRequest's` result_logits.
        Then we set the request to done, effectively simulating what would
        happen under the hood.

        Args:
            request (LlmInferenceExecRequest): Request that would be submitted to batcher.
        """
        result_logits = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
        data = [float(i) for i in range(math.prod(result_logits.shape))]
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()

    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback,
        decode_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=-1,
        max_completion_tokens=1,
    )

    # Single token generated
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array[0] == 15
        assert exec_req.input_token_ids[-1] == 15
        assert exec_req.start_position == 6


@pytest.mark.asyncio
async def test_greedy_decode_multiple_completions(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

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
        data[count] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback_multiple_completions,
        decode_callback=_batcher_callback_multiple_completions,
        results_callback=_results_callback,
        eos_token_id=-1,
        max_completion_tokens=5,
    )

    # Multiple tokens generated
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array == [0, 1, 2, 3, 4]
        assert len(exec_req.input_token_ids) == 11
        assert exec_req.start_position == 10


@pytest.mark.asyncio
async def test_greedy_decode_eos_token(
    device, exec_req: LlmInferenceExecRequest, greedy_token_selection_strategy
):
    results_array = []

    def _results_callback(token: int):
        results_array.append(token)

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
        data[count] = 16
        result_logits.items = data
        request.result_logits = result_logits
        request.done.set_success()
        count += 1

    exec_req.start_position = len(exec_req.input_token_ids) - 1
    decode_config = DecodeConfig(
        token_selection_strategy=TokenSelectionStrategy.GREEDY,
    )
    config = build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback_multiple_completions,
        decode_callback=_batcher_callback_multiple_completions,
        results_callback=_results_callback,
        eos_token_id=5,
        max_completion_tokens=10,
    )

    # Multiple tokens generated, eos is hit
    with patch.object(
        greedy_token_selection_strategy,
        "_token_selection_strategy_config",
        new=config,
    ):
        await greedy_token_selection_strategy.decode(exec_req)

        assert results_array == [0, 1, 2, 3, 4, 5]
        assert len(exec_req.input_token_ids) == 11
        assert exec_req.start_position == 10
