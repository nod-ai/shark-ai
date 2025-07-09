# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, List, Union

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
)

from .config import (
    DecodeConfig,
    get_strategy_from_str,
    TokenSelectionStrategy,
    TokenSelectionStrategyConfig,
)
from .scorer import BeamSearchScorer, DefaultScorer
from .token_selector import TokenSelector
from .sampler import Sampler


def build_token_selector_config(
    decode_config: DecodeConfig,
    prefill_batcher,
    decode_batcher,
    results_callback: Callable[[Union[int, List[int]]], None],
) -> TokenSelectionStrategyConfig:
    """Build a configuration class for a given token selection strategy.

    Args:
        token_selection_strategy (TokenSelectionStrategy): Strategy to use.
        prefill_callback (Callable[[LlmInferenceExecRequest], None]): Callback for invoking prefill. Typically a batcher function.
        decode_callback (Callable[[LlmInferenceExecRequest], None]): Callback for invoking decode. Typically a batcher function.
        results_callback (Callable[[Union[int, List[int]]], None]): Callback for during or after tokens are generated, depending on the strategy.
        eos_token_id (int): Token to stop generation on.
        max_completion_tokens (int): Max tokens to generate.

    Raises:
        NotImplementedError: Unsupported `TokenSelectionStrategy`.

    Returns:
        TokenSelectionStrategyConfig: Instantiated config for token selector.
    """
    return TokenSelectionStrategyConfig(
        decode_config,
        prefill_callback=prefill_batcher.submit,
        decode_callback=decode_batcher.submit,
        decode_begin_callback=decode_batcher.reserve_workitem,
        decode_end_callback=decode_batcher.complete_workitem,
        results_callback=results_callback,
    )


def build_token_selector(
    config: TokenSelectionStrategyConfig,
) -> BaseTokenSelectionStrategy:
    """Build a token selector, given a strategy and a config.

    Args:
        token_selection_strategy (TokenSelectionStrategy): Strategy to use.
        config (TokenSelectionStrategyConfig): Config containing necessary parameters for execution.

    Raises:
        NotImplementedError: Unsupported `TokenSelectionStrategy`.

    Returns:
        BaseTokenSelectionStrategy: Instantiated token selector. Either `IndependentTokenSelectionStrategy` or `BeamSearchTokenSelectionStrategy`.
    """
    scorer = (
        BeamSearchScorer(config=config)
        if config.decode_config.use_beam_search
        else DefaultScorer(config=config)
    )

    return TokenSelector(token_selection_strategy_config=config, scorer=scorer)


def is_multi_response(decode_config: DecodeConfig) -> bool:
    use_beam_search = decode_config.use_beam_search
    num_beams = decode_config.num_beams

    return use_beam_search or num_beams > 1


__all__ = [
    "build_token_selector",
    "build_token_selector_config",
    "BaseTokenSelectionStrategy",
    "BeamSearchScorer",
    "DefaultScorer",
    "get_strategy_from_str",
    "is_multi_response",
    "Sampler",
    "TokenSelectionStrategyConfig",
    "TokenSelectionStrategy",
    "TokenSelector",
]
