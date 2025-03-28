# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bisect
import logging

from typing import List, Set, Tuple

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from .beam_group import BeamGroup, ExecRequestSelection
from ..messages import LlmInferenceExecRequest, InferencePhase

from shortfin_apps.utils import convert_int_to_float

import shortfin.array as sfnp

logger = logging.getLogger(__name__)


class BeamSearchTokenSelectionStrategy(BaseTokenSelectionStrategy):
    def __init__(self, token_selection_strategy_config: TokenSelectionStrategyConfig):
        self._token_selection_strategy_config = token_selection_strategy_config

        self.min_log_prob = 0.0

    @property
    def token_selection_strategy_config(self):
        return self._token_selection_strategy_config

    def _top_k(
        self, log_softmax_logits: sfnp.device_array, k: int
    ) -> Tuple[List[int], List[float]]:
        """Select the top `k` tokens and values from the logits.

        Args:
            log_softmax_logits (sfnp.device_array): log_softmax of inference result_logits.
            k (int): Number of max elements to return.

        Returns:
            Tuple[List[int], List[float]]: Tuple containing (top_tokens, top_values)
        """
        partitioned_tokens = sfnp.argpartition(log_softmax_logits, k)
        top_tokens = partitioned_tokens.view(0, 0, slice(k, None)).items.tolist()
        values_view = log_softmax_logits.view(0, 0).items
        top_values = []
        for token in top_tokens:
            value = values_view[token]
            if isinstance(value, int):
                top_values.append(convert_int_to_float(value, log_softmax_logits.dtype))
            else:
                top_values.append(value)

        return top_tokens, top_values

    def _normalize_exec_req(
        self, exec_req: LlmInferenceExecRequest
    ) -> LlmInferenceExecRequest:
        """Accumulate the normalization of an LlmInferenceExecRequest.

        Args:
            exec_req (LlmInferenceExecRequest): Request to accumulate.

        Returns:
            LlmInferenceExecRequest: Request.
        """
        exec_req.accumulated_normalization += abs(self.min_log_prob)
        return exec_req

    def select_top_k(
        self,
        active_exec_reqs: List[LlmInferenceExecRequest],
        completed_exec_reqs: Set[LlmInferenceExecRequest],
    ) -> List[ExecRequestSelection]:
        """Handle the selection of the `top_k` beams within a decode step.

        Args:
            active_exec_reqs (List[LlmInferenceExecRequest]): Requests that are still active.
            completed_exec_reqs (Set[LlmInferenceExecRequest]): Requests that have been completed.

        Returns:
            List[ExecRequestSelection]: The `top_k` selections, containing necessary info for `beam_group` to handle choosing and processing beams.
        """
        config = self.token_selection_strategy_config
        k = config.decode_config.num_beams - len(completed_exec_reqs)

        global_min_log_prob = 0.0

        selections = []
        for exec_req in active_exec_reqs:
            # Take `log_softmax` of the logits.
            log_softmax_logits = sfnp.log_softmax(exec_req.result_logits)
            top_tokens, top_values = self._top_k(log_softmax_logits, -k)

            min_log_prob = 0.0
            for token, value in zip(top_tokens, top_values):
                if value < min_log_prob:
                    min_log_prob = value

                cumulative_log_prob = exec_req.score + value

                # Insert into sorted array
                selection = ExecRequestSelection(
                    exec_req,
                    token,
                    score=cumulative_log_prob,
                    normalization_function=self._normalize_exec_req,
                )
                index = bisect.bisect_left(
                    selections, selection.score, key=lambda x: x.score
                )
                selections.insert(index, selection)

                # Only maintain the `k` top scores
                if len(selections) > k:
                    selections.pop(0)

            if min_log_prob < global_min_log_prob:
                global_min_log_prob = min_log_prob

        self.min_log_prob = global_min_log_prob
        for selection in selections:
            selection.score -= global_min_log_prob
        selections = selections[::-1]
        return selections

    def _final_score(self, exec_req: LlmInferenceExecRequest) -> float:
        """Calculate the final score of a beam, post generation.

        Args:
            exec_req (LlmInferenceExecRequest): Request to calculate score of.

        Returns:
            float: Final score of a given beam.
        """
        return (exec_req.score - exec_req.accumulated_normalization) / (
            len(exec_req.input_token_ids) - exec_req.prompt_length
        )

    def _find_top_beam(
        self,
        active_reqs: List[LlmInferenceExecRequest],
        completed_reqs: Set[LlmInferenceExecRequest],
    ) -> LlmInferenceExecRequest:
        """Find the highest scoring beam, post generation.

        Args:
            active_reqs (List[LlmInferenceExecRequest]): Requests that are still actively generating.
            completed_reqs (Set[LlmInferenceExecRequest]): Requests that have completed.

        Returns:
            LlmInferenceExecRequest: Highest scoring request.
        """
        reqs = list(completed_reqs) if completed_reqs else active_reqs

        max_score = self._final_score(reqs[0])
        selected_req = reqs[0]
        for req in reqs[1:]:
            score = self._final_score(req)
            if score > max_score:
                selected_req = req
                max_score = score

        return selected_req

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `beam_search` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        config = self.token_selection_strategy_config

        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            [exec_req],
            self.select_top_k,
        )
        for _ in range(config.max_completion_tokens):
            if not beam_group.active_exec_reqs:
                break

            for req in beam_group.active_exec_reqs:
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)
            await beam_group.wait()
            beam_group.process_beams()

        self.get_results(beam_group)

    def get_results(self, beam_group: BeamGroup):
        """Get the results of a `beam_search` request, post generation.

        Args:
            beam_group (BeamGroup): Helper instance containing our beams.
        """
        config = self.token_selection_strategy_config
        results = [
            exec_req.input_token_ids[exec_req.prompt_length :]
            for exec_req in beam_group.completed_reqs
        ]
        if len(results) < beam_group.num_beams:
            active_exec_reqs = sorted(
                beam_group.active_exec_reqs,
                key=lambda exec_req: exec_req.score,
                reverse=True,
            )
            for i in range(beam_group.num_beams - len(results)):
                exec_req = active_exec_reqs[i]
                results.append(exec_req.input_token_ids[exec_req.prompt_length :])
        config.results_callback(results)
