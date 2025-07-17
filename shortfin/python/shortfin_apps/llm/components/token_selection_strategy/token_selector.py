# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import logging

from typing import List


from .beam_group import BeamGroup
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
)

from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


@dataclass
class TokenSelector(BaseTokenSelectionStrategy):
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

        exec_req.reset(InferencePhase.DECODE)

        beam_group = BeamGroup(
            exec_req,
            config.decode_config,
        )

        for _ in range(config.decode_config.max_completion_tokens - 1):
            if self.cancelled:
                break

            active_beam_count = len(beam_group.active_beams)
            config.decode_reserve_callback(
                rid=exec_req.orig_instance_id, count=active_beam_count
            )

            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)

            await beam_group.wait()
            beam_group.process_beams()

            if not beam_group.active_beams:
                break

        config.decode_reserve_callback(rid=exec_req.orig_instance_id, count=0)
        beam_group.clean_up()

        results = beam_group.get_results()
        config.results_callback(results)
