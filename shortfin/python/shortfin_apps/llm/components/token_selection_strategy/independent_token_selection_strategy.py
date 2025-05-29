# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import numpy as np
from typing import List

from .beam_group import Beam, BeamGroup
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
)

from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


TOP_P_DEFAULT_SELECTION = 32


class IndependentBeam(Beam):
    # TODO(stbaione): Combine this and `BeamSearchBeam` into a single class
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
        indices = exec_req.result_indices

        # Normal greedy selection based on max value
        if (top_k, top_p) == (None, None):
            if indices is not None:
                return indices.items[0]

            return self.sampler.select_greedy(logits)

        indices = np.array(indices) if indices is not None else None
        if top_k is not None:
            num_selections = 1 if top_p is None else top_k
            tokens, probs = self._sample_logits_top_k(
                logits,
                indices,
                top_k,
                num_selections,
            )

        if top_p is not None:
            if top_k is None:
                top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
                tokens, values = self.sampler.select_top_k(
                    logits, indices, -top_p_selection
                )
                probs = self._to_softmax(
                    values,
                    self.decode_config.logits_normalization,
                )

                if indices is None:
                    sorted_order = np.argsort(probs)[::-1]
                    tokens = tokens[sorted_order]
                    probs = probs[sorted_order]

            tokens, _ = self._sample_logits_top_p(tokens, probs, top_p, 1)

        return int(tokens[0])

    def update_exec_req(self):
        """Update the `LlmInferenceExecRequest` with the selected token."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def update_score(self, value):
        raise NotImplementedError("IndependentBeam does not track a score")

    def normalize_score(self, value):
        raise NotImplementedError("IndependentBeam does not track a score")

    def update_final_score(self):
        raise NotImplementedError("IndependentBeam does not track a score")


class IndependentTokenSelectionStrategy(BaseTokenSelectionStrategy):
    def select_greedy(
        self,
        active_beams: List[IndependentBeam],
        _: List[IndependentBeam],
    ) -> List[IndependentBeam]:
        """Greedily select a token for each active beam.

        Args:
            active_beams (List[IndependentBeam]): Beams that are still active.
            _ (List[IndependentBeam]): Beams that are completed.

        Returns:
            List[IndependentBeam]: Beams with new token selected.
        """
        selections = []
        for beam in active_beams:
            token = beam.sample_logits()
            beam.last_token = token
            selections.append(
                beam,
            )

        return selections

    def _stream_single_beam(self, beam_group: BeamGroup) -> List[IndependentBeam]:
        """Stream a single beam for the `multi_greedy` strategy.

        Args:
            beam_group (BeamGroup): The group of beams to process.

        Returns:
            List[IndependentBeam]: Beams with new token selected.
        """
        results_callback = self.token_selection_strategy_config.results_callback

        assert (
            beam_group.num_beams == 1
        ), "Streaming is not supported for multi-hypothesis yet."

        beam = beam_group.active_beams[0]
        results_callback(beam.last_token)

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `multi_greedy` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        self._log_sampling_method()
        config = self.token_selection_strategy_config

        exec_req.reset(InferencePhase.DECODE)

        num_beams = config.decode_config.num_beams

        # Copy `exec_req` to `num_beams` total requests
        if num_beams > 1:
            exec_reqs = self.replicate_inference_exec_requests(exec_req, num_beams - 1)
        else:
            exec_reqs = [exec_req]

        beams = [
            IndependentBeam(exec_req, decode_config=config.decode_config)
            for exec_req in exec_reqs
        ]
        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            beams,
            self.select_greedy,
        )

        reservations = beam_group.active_beam_count
        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if exec_req.status_tracker.is_disconnected():
                break
            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                release_amount = reservations - active_beam_count
                config.decode_end_callback(
                    rid=exec_req.orig_instance_id, count=release_amount
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

            if config.decode_config.num_beams == 1:
                self._stream_single_beam(beam_group)

        config.decode_end_callback(rid=exec_req.orig_instance_id, count=reservations)
        beam_group.clean_up()

        if config.decode_config.num_beams > 1:
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
