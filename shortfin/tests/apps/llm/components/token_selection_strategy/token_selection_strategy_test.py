# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import List
import pytest

import shortfin.array as sfnp

from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components import token_selection_strategy


def test_imports():
    for attr in token_selection_strategy.__all__:
        assert hasattr(token_selection_strategy, attr)


def _batcher_callback(exec_req: LlmInferenceExecRequest):
    pass


def _results_callback(token: int | List[int]):
    pass


def test_build_token_selector_config():
    strategy = token_selection_strategy.TokenSelectionStrategy.GREEDY
    decode_config = token_selection_strategy.DecodeConfig(
        token_selection_strategy=strategy,
    )

    config = token_selection_strategy.build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback,
        decode_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=0,
        max_completion_tokens=42,
    )

    assert config.prefill_callback == _batcher_callback
    assert config.decode_callback == _batcher_callback
    assert config.results_callback == _results_callback
    assert config.eos_token_id == 0
    assert config.max_completion_tokens == 42

    with pytest.raises(NotImplementedError):
        decode_config.token_selection_strategy = "NotImplemented"
        config = token_selection_strategy.build_token_selector_config(
            decode_config,
            prefill_callback=_batcher_callback,
            decode_callback=_batcher_callback,
            results_callback=_results_callback,
            eos_token_id=0,
            max_completion_tokens=42,
        )


def test_build_token_selector():
    strategy = token_selection_strategy.TokenSelectionStrategy.GREEDY

    decode_config = token_selection_strategy.DecodeConfig(
        token_selection_strategy=strategy,
    )
    config = token_selection_strategy.build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback,
        decode_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=0,
        max_completion_tokens=42,
    )
    token_selector = token_selection_strategy.build_token_selector(
        config,
    )
    assert token_selector._token_selection_strategy_config == config
    assert token_selector.token_selection_strategy_config == config

    with pytest.raises(NotImplementedError):
        config.decode_config.token_selection_strategy = "NotImplemented"
        token_selection_strategy.build_token_selector(config)


@pytest.mark.asyncio
async def test_prefill(
    device, exec_req: LlmInferenceExecRequest, dummy_token_selection_strategy
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

    strategy = token_selection_strategy.TokenSelectionStrategy.GREEDY

    decode_config = token_selection_strategy.DecodeConfig(
        token_selection_strategy=strategy,
    )

    config = token_selection_strategy.build_token_selector_config(
        decode_config,
        prefill_callback=_batcher_callback,
        decode_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=0,
        max_completion_tokens=1,
    )
    dummy_token_selection_strategy._token_selection_strategy_config = config
    await dummy_token_selection_strategy.prefill(exec_req)

    assert results_array[0] == 15
    assert exec_req.input_token_ids[-1] == 15
    assert exec_req.start_position == 6


def test_decode_config():
    num_beams = 42
    for strategy in [
        token_selection_strategy.TokenSelectionStrategy.GREEDY,
        token_selection_strategy.TokenSelectionStrategy.MULTI_GREEDY,
    ]:
        decode_config = token_selection_strategy.DecodeConfig(num_beams, strategy)
        assert decode_config.num_beams == 42
        assert decode_config.token_selection_strategy == strategy


def test_decode_config_str():
    # Str conversion at init
    num_beams = 42
    for strategy_str in ["greedy", "multi_greedy"]:
        decode_config = token_selection_strategy.DecodeConfig(num_beams, strategy_str)
        assert decode_config.num_beams == num_beams
        assert (
            decode_config.token_selection_strategy
            == token_selection_strategy.get_strategy_from_str(strategy_str)
        )
