# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np

from .beam_group import Beam
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


TOP_P_DEFAULT_SELECTION = 32


class GreedyBeam(Beam):
    def sample_logits(self) -> int:
        """Return the single highest scoring token of the logits.

        Returns:
            int: The `argmax` of the logits.
        """
        exec_req = self.exec_req
        decode_config = self.decode_config
        top_k = decode_config.top_k
        top_p = decode_config.top_p

        logits = np.array(exec_req.result_logits)

        # Normal greedy selection based on max value
        if (top_k, top_p) == (None, None):
            return self.sampler.select_greedy(logits)

        if top_k is not None:
            num_selections = 1 if top_p is None else top_k
            tokens, probs = self._sample_logits_top_k(
                logits,
                top_k,
                num_selections,
            )

        if top_p is not None:
            if top_k is None:
                top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
                tokens, values = self.sampler.select_top_k(logits, -top_p_selection)
                probs = self._to_softmax(
                    values,
                    self.decode_config.logits_normalization,
                )

                sorted_order = np.argsort(probs)[::-1]
                tokens = tokens[sorted_order]
                probs = probs[sorted_order]

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

        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=1)
        beam = GreedyBeam(exec_req, decode_config=config.decode_config)
        for _ in range(config.decode_config.max_completion_tokens):
            if exec_req.status_tracker.is_disconnected():
                break
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
        config.decode_end_callback(rid=exec_req.orig_instance_id, count=1)
        beam.exec_req.free_cache_pages()
