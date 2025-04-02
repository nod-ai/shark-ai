# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from abc import ABC, abstractmethod
from asyncio import gather
from dataclasses import dataclass
from typing import Callable, Dict, List, Set
from uuid import uuid4

from ..messages import LlmInferenceExecRequest


logger = logging.getLogger(__name__)


# TODO: Define `top_p` function in base class when enabled in
# shortfin.
@dataclass
class Beam(ABC):
    exec_req: LlmInferenceExecRequest

    score: float = 0.0
    accumulated_normalization: float = 0.0
    last_token: int | None = None

    @abstractmethod
    def sample_logits(self):
        pass

    @abstractmethod
    def update_score(self, value: float):
        pass

    @abstractmethod
    def update_exec_req(self):
        pass

    @abstractmethod
    def normalize_score(self, value: float):
        pass

    @abstractmethod
    def final_score(self, value: float):
        pass


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: List[Beam],
        selection_callback: Callable[
            [List[Beam], Set[Beam]],
            List[Beam],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams = beams
        self.selection_callback = selection_callback
        self.completed_beams: List[Beam] = []

    async def wait(self):
        done_signals = [beam.exec_req.done for beam in self.active_beams]
        return await gather(*done_signals)

    def process_beams(self):
        beam_selections = self.selection_callback(
            self.active_beams, self.completed_beams
        )
        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        active_beams: List[Beam] = []
        active_reqs: Set[LlmInferenceExecRequest] = set()
        completed_beams: List[Beam] = []
        completed_reqs: Set[LlmInferenceExecRequest] = set()

        for i in range(len(beam_selections)):
            beam = beam_selections[i]
            new_req, token = beam.exec_req, beam.last_token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                beam.exec_req = new_req

            visited_reqs[new_req.instance_id] = new_req
            if token == self.eos_token_id:
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
        for beam in self.active_beams:
            if beam.exec_req not in active_reqs and beam.exec_req not in completed_reqs:
                beam.exec_req.free_cache_pages()

        self.active_beams = active_beams
        self.completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up {self.beam_group_id}...")

        # Ensure all requests have freed their cache pages
        for beam in self.active_beams + list(self.completed_beams):
            beam.exec_req.free_cache_pages()
