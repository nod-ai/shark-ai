# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import List, Set

from .beam_group import BeamGroup, ExecRequestSelection
from .greedy_token_selection_strategy import GreedyTokenSelectionStrategy

from ..messages import LlmInferenceExecRequest, InferencePhase

import shortfin.array as sfnp

logger = logging.getLogger(__name__)


class MultiGreedyTokenSelectionStrategy(GreedyTokenSelectionStrategy):
    def select_greedy(
        self,
        active_exec_reqs: List[LlmInferenceExecRequest],
        _: Set[LlmInferenceExecRequest],
    ):
        selections = []
        for exec_req in active_exec_reqs:
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]
            selections.append(
                ExecRequestSelection(
                    exec_req,
                    token_int,
                )
            )

        return selections

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        config = self.token_selection_strategy_config

        exec_req.reset(InferencePhase.DECODE)

        # Copy `exec_req` to `num_beams` total requests
        exec_reqs = [exec_req]
        for _ in range(config.decode_config.num_beams - 1):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            exec_reqs,
            self.select_greedy,
        )

        for _ in range(config.max_completion_tokens):
            if not beam_group.active_exec_reqs:
                break
            for req in beam_group.active_exec_reqs:
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)
            await beam_group.wait()
            beam_group.process_beams()

        results = [
            exec_req.input_token_ids[exec_req.prompt_length :]
            for exec_req in beam_group.completed_reqs
        ]
        if len(results) < beam_group.num_beams:
            results.extend(
                [
                    exec_req.input_token_ids[exec_req.prompt_length :]
                    for exec_req in beam_group.active_exec_reqs
                ]
            )
        config.results_callback(results)
