# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List

from .config import TokenSelectionStrategyConfig
from .scorer import BaseBeamScorer

from ..messages import LlmInferenceExecRequest

import shortfin.array as sfnp


logger = logging.getLogger(__name__)


@dataclass
class BaseTokenSelectionStrategy(ABC):
    """Abstract class for implementing token selection strategies."""

    token_selection_strategy_config: TokenSelectionStrategyConfig
    scorer: BaseBeamScorer | None

    def _log_sampling_method(self):
        """Log the sampling method used for token selection."""
        decode_config = self.token_selection_strategy_config.decode_config
        num_beams = decode_config.num_beams
        strategy = "indepdent" if not decode_config.use_beam_search else "beam_search"
        logger.debug(f"Using {strategy} selection method with {num_beams} beams...")

        if decode_config.top_k is not None:
            logger.debug(
                f"Using `top_k` sampling with `top_k == {decode_config.top_k}`"
            )

        if decode_config.top_p is not None:
            logger.debug(
                f"Using `top_p` sampling with `top_p == {decode_config.top_p}`"
            )

    def replicate_inference_exec_requests(
        self, exec_req: LlmInferenceExecRequest, replicate: int
    ) -> List[LlmInferenceExecRequest]:
        """Replicate an LlmInferenceExecRequest for multi_beam strategies.

        Returns:
            List[LlmInferenceExecRequest]: List of replicated requests, including the original request.
        """
        exec_reqs = [exec_req]
        for _ in range(replicate):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

        return exec_reqs

    async def prefill(self, exec_req: LlmInferenceExecRequest) -> int:
        """Perform standard `prefill` on an LlmInferenceExecRequest.

        This takes an inference exec request and submits it to the batcher
        for prefill. We use greedy token selection to pick our 0th generated
        token.

        Args:
            exec_req (LlmInferenceExecRequest): Execution Request.

        Returns:
            int: Token generated from prefill.
        """

        if exec_req.status_tracker.is_disconnected():
            return

        token_selection_strategy_config = self.token_selection_strategy_config

        token_selection_strategy_config.prefill_callback(exec_req)
        await exec_req.done

        assert_message = f"{exec_req.instance_id}'s result_logits are None. This typically indicates an error during prefill invocation."
        assert exec_req.result_logits is not None, assert_message

        if exec_req.result_indices is not None:
            token_int = exec_req.result_indices.items[0]
        else:
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]

        exec_req.input_token_ids.append(token_int)
        exec_req.start_position = len(exec_req.input_token_ids) - 1

    @abstractmethod
    async def decode(self, exec_req: LlmInferenceExecRequest) -> List[int]:
        """Abstract method for generating completion tokens in a decode loop.

        This takes an LlmInferenceExecRequest, that has had prefill invoked
        on it. It then handles the logic of submitting decode requests to
        the batcher, in a loop, in order to generate a list of tokens.

        Args:
            exec_req (LlmInferenceExecRequest): Execution Request that has had prefill invoked on it.

        Returns:
            List[int]: A list of tokens generated from decode loop.
        """
        pass
