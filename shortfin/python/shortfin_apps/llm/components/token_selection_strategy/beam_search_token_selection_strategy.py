# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from typing import List

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from .beam_group import BeamGroup, Beam
from .config import LogitsNormalization
from ..messages import LlmInferenceExecRequest, InferencePhase

import shortfin.array as sfnp

from shortfin_apps.utils import (
    convert_float_to_int,
    convert_int_to_float,
    convert_list_to_device_array,
)


logger = logging.getLogger(__name__)


TOP_P_DEFAULT_SELECTION = 32


class BeamSearchBeam(Beam):
    def _convert_results_to_log_probs(
        self,
        probs: List,
    ):
        device = self.exec_req.result_logits.device
        dtype = self.exec_req.result_logits.dtype
        probs_sf = convert_list_to_device_array(
            probs,
            [len(probs)],
            device,
            dtype,
        )
        log_probs = self.convert_logits_normalization(
            LogitsNormalization.SOFTMAX,
            LogitsNormalization.LOG_SOFTMAX,
            probs_sf,
            **{"device_visible": True},
        )

        log_probs_dtype = log_probs.dtype

        if log_probs_dtype in [sfnp.float16]:
            log_probs = [
                convert_int_to_float(value, log_probs_dtype)
                for value in log_probs.items.tolist()
            ]
        else:
            log_probs = log_probs.items.tolist()

        return log_probs

    def sample_logits(self, k: int):
        """Obtain tokens and log_probs from beam_search or beam_search with sampling.

        Args:
            k (int): Number of max elements to return.

        Returns:
            Tuple[List[int], List[float]]: Tuple containing (top_tokens, top_values)
        """
        logits = self.exec_req.result_logits
        decode_config = self.decode_config
        num_beams = decode_config.num_beams
        top_k = decode_config.top_k
        top_p = decode_config.top_p

        if (top_k, top_p) == (None, None):
            tokens, probs = self.sampler.select_top_k(logits, -k)

            # TODO: https://github.com/nod-ai/shark-ai/issues/1278 find cleaner way to do these conversions
            if logits.dtype in [sfnp.float16]:
                probs = [convert_float_to_int(prob, logits.dtype) for prob in probs]

            probs_sf = convert_list_to_device_array(
                probs,
                [len(probs)],
                logits.device,
                logits.dtype,
            )

            if self.decode_config.logits_normalization == LogitsNormalization.NONE:
                probs_sf = self.apply_temperature(probs_sf)

            log_probs = self.convert_logits_normalization(
                self.decode_config.logits_normalization,
                LogitsNormalization.LOG_SOFTMAX,
                probs_sf,
            ).items.tolist()

            if logits.dtype in [sfnp.float16]:
                log_probs = [
                    convert_int_to_float(log_prob, logits.dtype)
                    for log_prob in log_probs
                ]

            return tokens, log_probs

        if top_k is not None:
            # Sample from `top_k` tokens
            tokens, probs = self._sample_logits_top_k(
                logits,
                top_k,
                num_selections=top_k,
            )

        if top_p is not None:
            if top_k is None:
                top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
                tokens, values = self.sampler.select_top_k(logits, -top_p_selection)
                probs = self._to_softmax(
                    values,
                    logits.dtype,
                    logits.device,
                    self.decode_config.logits_normalization,
                )

            tokens, probs = self._sample_logits_top_p(tokens, probs, top_p, num_beams)

        if logits.dtype in [sfnp.float16]:
            probs = [convert_float_to_int(prob, logits.dtype) for prob in probs]

        log_probs = self._convert_results_to_log_probs(
            probs,
        )

        return tokens, log_probs

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

        top_score = None
        top_beam = None
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
                    decode_config=config.decode_config,
                )
                new_beam.update_score(value)
                selections.append(new_beam)

                if top_score is None or new_beam.score > top_score:
                    top_score = new_beam.score
                    top_beam = new_beam

            if min_log_prob < global_min_log_prob:
                global_min_log_prob = min_log_prob

        if len(selections) < config.decode_config.num_beams:
            beams_to_add = config.decode_config.num_beams - len(selections)
            for _ in range(beams_to_add):
                new_beam = BeamSearchBeam(
                    exec_req=top_beam.exec_req,
                    score=top_beam.score,
                    accumulated_normalization=top_beam.accumulated_normalization,
                    last_token=top_beam.last_token,
                    decode_config=config.decode_config,
                )
                selections.append(new_beam)

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
        self._log_sampling_method()
        config = self.token_selection_strategy_config

        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            [BeamSearchBeam(exec_req, decode_config=config.decode_config)],
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
