# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from asyncio import gather
from dataclasses import dataclass
from typing import Callable, Dict, List, Set
from uuid import uuid4

from ..messages import LlmInferenceExecRequest


logger = logging.getLogger(__name__)


@dataclass
class ExecRequestSelection:
    """Helper class to standardize the return"""

    exec_req: LlmInferenceExecRequest
    token: int
    score: float | None = None
    normalization_function: (
        None | Callable[[LlmInferenceExecRequest], LlmInferenceExecRequest]
    ) = None


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        exec_reqs: List[LlmInferenceExecRequest],
        selection_callback: Callable[
            [List[LlmInferenceExecRequest], Set[LlmInferenceExecRequest]],
            List[ExecRequestSelection],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_exec_reqs = exec_reqs
        self.selection_callback = selection_callback
        self.completed_reqs: set[LlmInferenceExecRequest] = set()

    async def wait(self):
        done_signals = [req.done for req in self.active_exec_reqs]
        return await gather(*done_signals)

    def _update_exec_req(self, selection: ExecRequestSelection):
        token = selection.token
        exec_req = selection.exec_req

        exec_req.input_token_ids.append(token)
        exec_req.start_position += 1
        if selection.normalization_function is not None:
            exec_req = selection.normalization_function(exec_req)
        if selection.score is not None:
            exec_req.score = selection.score

    def process_beams(self):
        exec_reqs_selections = self.selection_callback(
            self.active_exec_reqs, self.completed_reqs
        )
        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        active_reqs = set()
        completed_reqs = set()
        req_selection_map: Dict[str, ExecRequestSelection] = {}

        for selection in exec_reqs_selections:
            new_req, token = selection.exec_req, selection.token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                selection.exec_req = new_req

            req_selection_map[new_req.instance_id] = selection
            visited_reqs[new_req.instance_id] = new_req
            if token == self.eos_token_id:
                completed_reqs.add(new_req)
            else:
                active_reqs.add(new_req)

        for req in completed_reqs | active_reqs:
            self._update_exec_req(req_selection_map[req.instance_id])
            if req in completed_reqs:
                req.free_cache_pages()

        # Free cache pages of reqs we don't need anymore
        for req in self.active_exec_reqs:
            if req not in active_reqs and req not in completed_reqs:
                req.free_cache_pages()

        self.active_exec_reqs = list(active_reqs)
        self.completed_reqs |= completed_reqs

    def clean_up(self):
        logger.debug(f"Cleaning up {self.beam_group_id}...")

        # Ensure all requests have freed their cache pages
        for req in self.active_exec_reqs + list(self.completed_reqs):
            req.free_cache_pages()
