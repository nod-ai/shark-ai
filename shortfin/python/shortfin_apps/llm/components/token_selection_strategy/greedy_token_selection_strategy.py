# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import shortfin.array as sfnp

from .beam_group import Beam, LogitsNormalization
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from ..io_struct import NOT_PROVIDED
from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


class GreedyBeam(Beam):
    def sample_logits(self) -> int:
        """Return the single highest scoring token of the logits.

        Returns:
            int: The `argmax` of the logits.
        """
        self.apply_temperature()
        exec_req = self.exec_req
        decode_config = self.decode_config
        top_k = decode_config.top_k
        top_p = decode_config.top_p

        # Normal greedy selection based on max value
        if (top_k, top_p) == (NOT_PROVIDED, NOT_PROVIDED):
            return self.sampler.select_greedy(exec_req.result_logits)

        # Convert to softmax to obtain probabilities
        softmax_logits = self.convert_logits_normalization(
            decode_config.logits_normalization,
            LogitsNormalization.SOFTMAX,
            exec_req.result_logits,
        )

        tokens, probs = softmax_logits, None
        if top_k != NOT_PROVIDED:
            num_selections = 1 if top_p == NOT_PROVIDED else top_k
            tokens, probs = self._sample_logits_top_k(
                softmax_logits,
                top_k,
                num_selections,
            )

        if top_p != NOT_PROVIDED:
            if top_k == NOT_PROVIDED:
                tokens, probs = self.sampler.select_top_k(tokens, -32)
            tokens, _ = self._sample_logits_top_p(tokens, probs, top_p, 1)

        return tokens[0]

    def update_exec_req(self):
        """Update the `LlmInferenceExecRequest` with the selected token."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def update_score(self, value):
        raise NotImplementedError("GreedyBeam does not track a score")

    def normalize_score(self, value):
        raise NotImplementedError("GreedyBeam does not track a score")

    def update_final_score(self):
        raise NotImplementedError("GreedyBeam does not track a score")


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
        self._log_sampling_method()
        config = self.token_selection_strategy_config

        config.decode_begin_callback(1)
        beam = GreedyBeam(exec_req, decode_config=config.decode_config)
        for _ in range(config.decode_config.max_completion_tokens):
            exec_req = beam.exec_req
            exec_req.reset(InferencePhase.DECODE)
            config.decode_callback(exec_req)
            await exec_req.done
            token_int = beam.sample_logits()
            beam.last_token = token_int
            config.results_callback(token_int)
            if token_int == config.eos_token_id:
                break
            beam.update_exec_req()
        config.decode_end_callback(1)
        beam.exec_req.free_cache_pages()
