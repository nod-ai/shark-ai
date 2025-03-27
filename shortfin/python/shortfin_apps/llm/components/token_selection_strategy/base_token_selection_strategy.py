# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Callable, Union

from dataclasses_json import dataclass_json, Undefined

from ..messages import LlmInferenceExecRequest

import shortfin.array as sfnp


class TokenSelectionStrategy(Enum):
    """Supported token selection strategies."""

    GREEDY = auto()
    MULTI_GREEDY = auto()


def get_strategy_from_str(token_selection_strategy: str) -> TokenSelectionStrategy:
    name_to_strategy = {
        strategy.name.lower(): strategy for strategy in TokenSelectionStrategy
    }
    strategy = token_selection_strategy.lower()
    if strategy not in name_to_strategy:
        raise KeyError(f"Unknown token_selection_strategy: {token_selection_strategy}")

    return name_to_strategy[strategy]


def is_ref_counted(token_selection_strategy: TokenSelectionStrategy) -> bool:
    return token_selection_strategy in {TokenSelectionStrategy.MULTI_GREEDY}


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DecodeConfig:

    # Number of beams to use during generation
    num_beams: int = 1

    # Strategy for selecting tokens during generation
    token_selection_strategy: str | TokenSelectionStrategy = "greedy"

    def __post_init__(self):
        if isinstance(self.token_selection_strategy, str):
            self.token_selection_strategy = get_strategy_from_str(
                self.token_selection_strategy
            )


@dataclass
class TokenSelectionStrategyConfig:
    """Configuration for token selection strategies."""

    decode_config: DecodeConfig
    prefill_callback: Callable[[LlmInferenceExecRequest], None]
    decode_callback: Callable[[LlmInferenceExecRequest], None]
    decode_begin_callback: Callable[[], None]
    decode_end_callback: Callable[[], None]
    results_callback: Callable[[Union[int, List[int]]], None]
    eos_token_id: int
    max_completion_tokens: int


class BaseTokenSelectionStrategy(ABC):
    """Abstract class for implementing token selection strategies."""

    @property
    @abstractmethod
    def token_selection_strategy_config(self) -> TokenSelectionStrategyConfig:
        """Configuration object for defining the parameters of the decode loop.

        Returns:
            TokenSelectionStrategyConfig: Configuration object.
        """
        pass

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
        token_selection_strategy_config = self.token_selection_strategy_config

        token_selection_strategy_config.prefill_callback(exec_req)
        await exec_req.done

        token = sfnp.argmax(exec_req.result_logits)
        token_int = token.items[0]
        decode_config = token_selection_strategy_config.decode_config
        # TODO: This is only temporary until streaming is enabled for `MultiGreedy`
        if decode_config.token_selection_strategy == TokenSelectionStrategy.GREEDY:
            token_selection_strategy_config.results_callback(token_int)

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
