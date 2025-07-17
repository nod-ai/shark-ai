# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from asyncio import gather
from typing import Dict, List, Set, Optional
from uuid import uuid4

from .config import DecodeConfig
from ..messages import LlmInferenceExecRequest
from .scorer import BeamSearchScorer, DefaultScorer
from .beams import BaseBeam, BeamSearchBeam, DefaultBeam

logger = logging.getLogger(__name__)

TOP_P_DEFAULT_SELECTION = 32


class BeamGroup:
    def __init__(
        self,
        exec_req: LlmInferenceExecRequest,
        decode_config: DecodeConfig,
        beams: Optional[List[BaseBeam]] = None,
    ):
        exec_reqs = [exec_req]

        if beams is None:
            if not decode_config.use_beam_search and decode_config.num_beams > 1:
                for _ in range(decode_config.num_beams - 1):
                    exec_reqs.append(
                        LlmInferenceExecRequest.copy_exec_request(exec_req)
                    )

            beam_class = (
                BeamSearchBeam if decode_config.use_beam_search else DefaultBeam
            )
            self._active_beams = [
                beam_class(exec_req, decode_config=decode_config)
                for exec_req in exec_reqs
            ]
        else:
            self._active_beams = beams

        self._beam_group_id = str(uuid4())
        self._eos_token_id = decode_config.eos_token_id
        self._num_beams = decode_config.num_beams
        self._scorer = (
            BeamSearchScorer(decode_config.num_beams)
            if decode_config.use_beam_search
            else DefaultScorer(decode_config.num_beams)
        )
        self._completed_beams: List[BaseBeam] = []

    @property
    def active_beam_count(self):
        return len(self._active_beams)

    @property
    def active_beams(self):
        return self._active_beams

    @property
    def completed_beam_count(self):
        return len(self._completed_beams)

    # TODO(@zeehanhaque21): If the strategy is not beam search,
    # we can probably wait on each beam separately.
    async def wait(self):
        done_signals = [beam.exec_req.done for beam in self._active_beams]
        return await gather(*done_signals)

    def process_beams(self):
        beam_selections = self._scorer.select_beams(
            self._active_beams, self._completed_beams
        )

        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        active_beams: List[BaseBeam] = []
        active_reqs: Set[LlmInferenceExecRequest] = set()
        completed_beams: List[BaseBeam] = []
        completed_reqs: Set[LlmInferenceExecRequest] = set()

        for i in range(len(beam_selections)):
            beam = beam_selections[i]
            new_req, token = beam.exec_req, beam.last_token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                beam.exec_req = new_req

            visited_reqs[new_req.instance_id] = new_req
            if token == self._eos_token_id:
                completed_beams.append(beam)
                completed_reqs.add(new_req)
            else:
                active_beams.append(beam)
                active_reqs.add(new_req)

        for beam in completed_beams + active_beams:
            beam.update_exec_req()
            if beam.exec_req in completed_reqs:
                beam.exec_req.free_cache_pages()

        # Free cache pages of reqs we don't need anymore
        for beam in self._active_beams:
            if beam.exec_req not in active_reqs and beam.exec_req not in completed_reqs:
                beam.exec_req.free_cache_pages()

        self._active_beams = active_beams
        self._completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up {self._beam_group_id}...")

        # Ensure all requests have freed their cache pages
        for beam in self._active_beams + self._completed_beams:
            beam.exec_req.free_cache_pages()

    def _get_results_beam_search(self, results: List[List[int]]):
        for beam in self._active_beams:
            self._scorer.finalize_score(beam)

        active_beams = self._scorer.score_beams(
            self._active_beams, len(self._active_beams), normalize=False
        )
        for i in range(self._num_beams - len(results)):
            beam = active_beams[i]
            results.append(beam.exec_req.input_token_ids[beam.exec_req.prompt_length :])

        return results

    def get_results(self) -> List[List[int]]:

        logger.debug(f"Active beams: {len(self._active_beams)}")
        logger.debug(f"Completed beams: {len(self._completed_beams)}")
        logger.debug(f"Num beams: {self._num_beams}")
        # Get results from completed beams
        results = [
            beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
            for beam in self._completed_beams
        ]
        # If there are less than num_beams results, get results from active beams
        if len(results) < self._num_beams:
            results = self._get_results_beam_search(results)

        return results
