# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from typing import List, Set, Tuple

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from .beam_group import BeamGroup, Beam
from .config import LogitsNormalization
from ..messages import LlmInferenceExecRequest, InferencePhase

from shortfin_apps.utils import convert_int_to_float

import shortfin.array as sfnp

logger = logging.getLogger(__name__)


class BeamSearchBeam(Beam):
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

    def sample_logits(self, k: int):
        """Apply `log_softmax` and take the `top_k` token and values of the logits.

        Args:
            k (int): Number of max elements to return.

        Returns:
            Tuple[List[int], List[float]]: Tuple containing (top_tokens, top_values)
        """
        self.apply_temperature()
        if self.logits_normalization == LogitsNormalization.LOG_SOFTMAX:
            log_softmax_logits = self.exec_req.result_logits
        elif self.logits_normalization == LogitsNormalization.NONE:
            log_softmax_logits = sfnp.log_softmax(self.exec_req.result_logits)
        elif self.logits_normalization == LogitsNormalization.SOFTMAX:
            log_softmax_logits = sfnp.log(self.exec_req.result_logits)
        else:
            raise ValueError(
                f"Unknown logits normalization {self.logits_normalization}"
            )
        return self._top_k(log_softmax_logits, -k)

    def update_score(self, log_prob: float):
        """Increment the cumulative_log_prob of the beam.

        Args:
            log_prob (float): Log probability of the token.
        """
        self.score += log_prob

    def update_exec_req(self):
        """Add a selected token to a request after a decode loop."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def normalize_score(self, min_log_prob: float):
        """Track the accumulated_normalization for a given beam.

        Args:
            min_log_prob (float): Minimum log probability of the selected tokens.
        """
        self.accumulated_normalization += abs(min_log_prob)

    def update_final_score(self):
        """Calculate the final score of a beam, with a brevity penalty."""
        exec_req = self.exec_req
        self.score = (self.score - self.accumulated_normalization) / (
            len(exec_req.input_token_ids) - exec_req.prompt_length
        )


class BeamSearchTokenSelectionStrategy(BaseTokenSelectionStrategy):
    def __init__(self, token_selection_strategy_config: TokenSelectionStrategyConfig):
        self._token_selection_strategy_config = token_selection_strategy_config

        self.min_log_prob = 0.0

    @property
    def token_selection_strategy_config(self):
        return self._token_selection_strategy_config

    def select_top_k(
        self,
        active_beams: List[BeamSearchBeam],
        completed_beams: List[BeamSearchBeam],
    ) -> List[BeamSearchBeam]:
        """Handle the selection of the `top_k` beams within a decode step.

        Args:
            active_beams (List[BeamSearchBeam]): Beams that are still active.
            completed_beams (Set[BeamSearchBeam]): Beams that have been completed.

        Returns:
            List[BeamSearchBeam]: The `top_k` selections, containing necessary info for `beam_group` to handle choosing and processing beams.
        """
        config = self.token_selection_strategy_config
        k = config.decode_config.num_beams - len(completed_beams)

        global_min_log_prob = 0.0

        selections: List[BeamSearchBeam] = []
        for beam in active_beams:
            min_log_prob = 0.0
            top_tokens, top_values = beam.sample_logits(k)
            for token, value in zip(top_tokens, top_values):
                if value < min_log_prob:
                    min_log_prob = value

                new_beam = BeamSearchBeam(
                    exec_req=beam.exec_req,
                    score=beam.score,
                    accumulated_normalization=beam.accumulated_normalization,
                    last_token=token,
                    temperature=config.decode_config.temperature,
                    logits_normalization=config.decode_config.logits_normalization,
                )
                new_beam.update_score(value)
                selections.append(new_beam)

            if min_log_prob < global_min_log_prob:
                global_min_log_prob = min_log_prob

        sorted_selections = sorted(
            selections, key=lambda beam: beam.score, reverse=True
        )[:k]
        for beam in sorted_selections:
            beam.normalize_score(global_min_log_prob)
        return sorted_selections

    def _find_top_beam(
        self,
        active_beams: List[BeamSearchBeam],
        completed_beams: List[BeamSearchBeam],
    ) -> BeamSearchBeam:
        """Find the highest scoring beam, post generation.

        Args:
            active_beams (List[BeamSearchBeam]): Beams that are still actively generating.
            completed_beams (List[BeamSearchBeam]): Beams that have completed.

        Returns:
            BeamSearchBeam: Highest scoring beam.
        """
        beams = list(completed_beams) if completed_beams else active_beams
        for beam in beams:
            beam.update_final_score()
        return max(beams, key=lambda beam: beam.score)

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `beam_search` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        logger.info("Starting `beam_search` decode loop...")
        config = self.token_selection_strategy_config

        beam = BeamSearchBeam(
            exec_req=exec_req,
            temperature=config.decode_config.temperature,
            logits_normalization=config.decode_config.logits_normalization,
        )
        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            [beam],
            self.select_top_k,
        )

        reservations = beam_group.active_beam_count
        config.decode_begin_callback(reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if not beam_group.active_beams:
                break

            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                config.decode_end_callback(reservations - active_beam_count)
                reservations = active_beam_count

            if reservations < active_beam_count:
                config.decode_begin_callback(active_beam_count - reservations)
                reservations = active_beam_count

            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)
            await beam_group.wait()
            beam_group.process_beams()

        config.decode_end_callback(reservations)
        beam_group.clean_up()
        self.get_results(beam_group)

    def get_results(self, beam_group: BeamGroup):
        """Get the results of a `beam_search` request, post generation.

        Args:
            beam_group (BeamGroup): Helper instance containing our beams.
        """
        config = self.token_selection_strategy_config
        results = [
            beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            for beam in beam_group.active_beams:
                beam.update_final_score()

            active_beams = sorted(
                [beam for beam in beam_group.active_beams],
                key=lambda beam: beam.score,
                reverse=True,
            )
            for i in range(beam_group.num_beams - len(results)):
                beam = active_beams[i]
                results.append(
                    beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
                )
        config.results_callback(results)
