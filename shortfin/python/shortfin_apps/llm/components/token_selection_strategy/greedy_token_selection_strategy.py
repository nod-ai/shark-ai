# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import shortfin.array as sfnp

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from ..messages import LlmInferenceExecRequest, InferencePhase


logger = logging.getLogger(__name__)


class GreedyTokenSelectionStrategy(BaseTokenSelectionStrategy):
    def __init__(
        self,
        token_selection_strategy_config: TokenSelectionStrategyConfig,
    ):
        self._token_selection_strategy_config = token_selection_strategy_config

    @property
    def token_selection_strategy_config(self):
        return self._token_selection_strategy_config

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Perform greedy token selection in a loop, to obtain decode token list.

        Args:
            exec_req (LlmInferenceExecRequest): Execution request that has had prefill invoked on it.
        """
        config = self.token_selection_strategy_config
        for _ in range(config.max_completion_tokens):
            exec_req.reset(InferencePhase.DECODE)
            config.decode_callback(exec_req)
            await exec_req.done
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]
            config.results_callback(token_int)
            if token_int == config.eos_token_id:
                break
            exec_req.input_token_ids.append(token_int)
            exec_req.start_position += 1
