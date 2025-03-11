# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import math
import pytest
from typing import Generator
from unittest.mock import patch

import shortfin.array as sfnp

from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from shortfin_apps.llm.components import decode_strategy

from uuid import uuid4

logger = logging.getLogger(__name__)


class MockVoidFuture:
    def __init__(self):
        self._event = asyncio.Event()

    def set_success(self):
        self._event.set()

    def __await__(self):
        return self._event.wait().__await__()


@pytest.fixture(scope="function")
def exec_req():
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        yield LlmInferenceExecRequest(
            phase=InferencePhase.PREFILL,
            input_token_ids=[0, 1, 2, 3, 4, 5],
            rid=str(uuid4),
        )


class DummyDecodeStrategy(decode_strategy.DecodeStrategy):
    def __init__(self, decode_strategy_config: decode_strategy.DecodeStrategyConfig):
        # Initialize with a dummy config instance.
        self._decode_strategy_config = decode_strategy_config

    @property
    def decode_strategy_config(self):
        return self._decode_strategy_config

    async def decode(self, exec_req):
        pass


@pytest.fixture(scope="function")
def dummy_decode_strategy() -> decode_strategy.DecodeStrategy:
    return DummyDecodeStrategy(
        None,
    )


def test_imports():
    for attr in decode_strategy.__all__:
        assert hasattr(decode_strategy, attr)


@pytest.mark.asyncio
async def test_prefill(
    device, exec_req: LlmInferenceExecRequest, dummy_decode_strategy
):
    def _batcher_callback(request: LlmInferenceExecRequest):
        """Mock the batcher function to isolate `DecodeStrategy.prefill`.

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

    decode_strategy_config = decode_strategy.DecodeStrategyConfig(
        batcher_callback=_batcher_callback,
        results_callback=_results_callback,
        eos_token_id=0,
        max_completion_tokens=1,
    )
    dummy_decode_strategy._decode_strategy_config = decode_strategy_config
    await dummy_decode_strategy.prefill(exec_req)

    assert results_array[0] == 15
    assert exec_req.input_token_ids[-1] == 15
    assert exec_req.start_position == 6
