# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import logging

from typing import List


from .beam_group import BeamGroup, build_beam_group
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
)
from .scorer import BeamSearchScorer, DefaultScorer

from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


@dataclass
class TokenSelector(BaseTokenSelectionStrategy):
    scorer: BeamSearchScorer | DefaultScorer
    min_log_prob: float = 0.0

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `multi_greedy` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        if self.cancelled:
            return

        self._log_sampling_method()

        config = self.token_selection_strategy_config
        use_beam_search = config.decode_config.use_beam_search

        exec_req.reset(InferencePhase.DECODE)

        beam_group = build_beam_group(
            exec_req,
            config,
            self.scorer.select_beams,
        )

        reservations = beam_group.active_beam_count
        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if self.cancelled:
                break

            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                release_amount = reservations - active_beam_count
                config.decode_end_callback(
                    rid=exec_req.orig_instance_id, count=release_amount
                )
                reservations = active_beam_count

            if reservations < active_beam_count:
                acquire_amount = active_beam_count - reservations
                config.decode_begin_callback(
                    rid=exec_req.orig_instance_id, count=acquire_amount
                )
                reservations = active_beam_count

            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)

            await beam_group.wait()
            beam_group.process_beams()

            if not beam_group.active_beams:
                break

        config.decode_end_callback(rid=exec_req.orig_instance_id, count=reservations)
        beam_group.clean_up()

        self.get_results(beam_group)

    def _get_results_beam_search(self, beam_group: BeamGroup, results: List[List[int]]):
        for beam in beam_group.active_beams:
            self.scorer.finalize_score(beam)

        active_beams = sorted(
            [beam for beam in beam_group.active_beams],
            key=lambda beam: beam.score,
            reverse=True,
        )
        active_beams = beam_group.active_beams
        active_beams = self.scorer.score_beams(
            active_beams, len(active_beams), normalize=False
        )
        for i in range(beam_group.num_beams - len(results)):
            beam = active_beams[i]
            results.append(beam.exec_req.input_token_ids[beam.exec_req.prompt_length :])

        return results

    def get_results(self, beam_group: BeamGroup):
        config = self.token_selection_strategy_config

        results = [
            beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            results = self._get_results_beam_search(beam_group, results)

        config.results_callback(results)
