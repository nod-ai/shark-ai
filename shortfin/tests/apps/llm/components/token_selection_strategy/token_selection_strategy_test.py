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


class DummyTokenSelectionStrategy(token_selection_strategy.BaseTokenSelectionStrategy):
    async def decode(self, exec_req):
        pass


class FakeBatcher:
    def __init__(self, submit_cb, workitem_cb):
        self.submit = submit_cb
        self.reserve_workload = workitem_cb


def _batcher_workitem_callback():
    pass


def test_imports():
    for attr in token_selection_strategy.__all__:
        assert hasattr(token_selection_strategy, attr)


def _batcher_callback(exec_req: LlmInferenceExecRequest):
    pass


def _results_callback(token: int | List[int]):
    pass


def test_build_token_selector_config():
    decode_config = token_selection_strategy.DecodeConfig(
        max_completion_tokens=42,
        eos_token_id=0,
    )

    config = token_selection_strategy.build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        decode_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        results_callback=_results_callback,
    )

    assert config.prefill_callback == _batcher_callback
    assert config.decode_callback == _batcher_callback
    assert config.results_callback == _results_callback
    assert config.decode_config.eos_token_id == 0
    assert config.decode_config.max_completion_tokens == 42


@pytest.mark.asyncio
async def test_prefill(
    device,
    exec_req: LlmInferenceExecRequest,
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

    decode_config = token_selection_strategy.DecodeConfig(
        max_completion_tokens=1,
        eos_token_id=0,
    )

    config = token_selection_strategy.build_token_selector_config(
        decode_config,
        prefill_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        decode_batcher=FakeBatcher(_batcher_callback, _batcher_workitem_callback),
        results_callback=_results_callback,
    )
    dummy_token_selection_strategy = DummyTokenSelectionStrategy(
        token_selection_strategy_config=config,
    )
    await dummy_token_selection_strategy.prefill(exec_req)

    assert exec_req.input_token_ids[-1] == 15
    assert exec_req.start_position == 6


def test_decode_config():
    num_beams = 42
    decode_config = token_selection_strategy.DecodeConfig(
        num_beams=num_beams, eos_token_id=0
    )
    assert decode_config.num_beams == 42
    assert not decode_config.use_beam_search

    decode_config = token_selection_strategy.DecodeConfig(
        eos_token_id=0,
        num_beams=num_beams,
        use_beam_search=True,
    )

    assert decode_config.num_beams == 42
    assert decode_config.use_beam_search
