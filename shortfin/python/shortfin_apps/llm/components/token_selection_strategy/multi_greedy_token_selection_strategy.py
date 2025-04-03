# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import List, Set

from .beam_group import BeamGroup, Beam
from .greedy_token_selection_strategy import GreedyTokenSelectionStrategy

from ..messages import LlmInferenceExecRequest, InferencePhase

import shortfin.array as sfnp

logger = logging.getLogger(__name__)


class MultiGreedyBeam(Beam):
    def sample_logits(self) -> int:
        """Return the single highest scoring token of the logits.

        Returns:
            int: The `argmax` of the logits.
        """
        exec_req = self.exec_req
        token = sfnp.argmax(exec_req.result_logits)
        token_int = token.items[0]
        return token_int

    def update_exec_req(self):
        """Update the `LlmInferenceExecRequest` with the selected token."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def update_score(self, value):
        raise NotImplementedError("MultiGreedyBeam does not track a score")

    def normalize_score(self, value):
        raise NotImplementedError("MultiGreedyBeam does not track a score")

    def update_final_score(self):
        raise NotImplementedError("MultiGreedyBeam does not track a score")


class MultiGreedyTokenSelectionStrategy(GreedyTokenSelectionStrategy):
    def select_greedy(
        self,
        active_beams: List[MultiGreedyBeam],
        _: List[MultiGreedyBeam],
    ) -> List[MultiGreedyBeam]:
        """Greedily select a token for each active beam.

        Args:
            active_beams (List[MultiGreedyBeam]): Beams that are still active.
            _ (List[MultiGreedyBeam]): Beams that are completed.

        Returns:
            List[MultiGreedyBeam]: Beams with new token selected.
        """
        selections = []
        for beam in active_beams:
            token = beam.sample_logits()
            beam.last_token = token
            selections.append(
                beam,
            )

        return selections

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `multi_greedy` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        config = self.token_selection_strategy_config

        exec_req.reset(InferencePhase.DECODE)

        # Copy `exec_req` to `num_beams` total requests
        exec_reqs = self.replicate_inference_exec_requests(
            exec_req, config.decode_config.num_beams - 1
        )

        beams = [MultiGreedyBeam(exec_req) for exec_req in exec_reqs]
        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            beams,
            self.select_greedy,
        )

        for _ in range(config.max_completion_tokens):
            if not beam_group.active_beams:
                break
            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)
            await beam_group.wait()
            beam_group.process_beams()

        results = [
            beam.exec_req.input_token_ids[exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            results.extend(
                [
                    beam.exec_req.input_token_ids[exec_req.prompt_length :]
                    for beam in beam_group.active_beams
                ]
            )
        config.results_callback(results)
