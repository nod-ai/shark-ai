# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, List, Union

from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    DecodeConfig,
    TokenSelectionStrategyConfig,
    TokenSelectionStrategy,
    get_strategy_from_str,
    is_ref_counted,
)
from .greedy_token_selection_strategy import GreedyTokenSelectionStrategy
from .multi_greedy_token_selection_strategy import MultiGreedyTokenSelectionStrategy

from ..messages import LlmInferenceExecRequest


def build_token_selector_config(
    decode_config: DecodeConfig,
    prefill_batcher,
    decode_batcher,
    results_callback: Callable[[Union[int, List[int]]], None],
    eos_token_id: int,
    max_completion_tokens: int,
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
    config: None | TokenSelectionStrategyConfig = None
    match decode_config.token_selection_strategy:
        case TokenSelectionStrategy.GREEDY | TokenSelectionStrategy.MULTI_GREEDY:
            config = TokenSelectionStrategyConfig(
                decode_config,
                prefill_callback=prefill_batcher.submit,
                decode_callback=decode_batcher.submit,
                decode_begin_callback=decode_batcher.reserve_workitem,
                decode_end_callback=decode_batcher.complete_workitem,
                results_callback=results_callback,
                eos_token_id=eos_token_id,
                max_completion_tokens=max_completion_tokens,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported token selection strategy: {decode_config.token_selection_strategy}.\n"
                f"Supported strategies: {','.join([strategy.name for strategy in TokenSelectionStrategy])}"
            )
    return config


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
        BaseTokenSelectionStrategy: Instantiated token selector. Current only `Greedy`, but more will be added.
    """
    token_selector: BaseTokenSelectionStrategy | None = None
    match config.decode_config.token_selection_strategy:
        case TokenSelectionStrategy.GREEDY:
            token_selector = GreedyTokenSelectionStrategy(
                config,
            )
        case TokenSelectionStrategy.MULTI_GREEDY:
            token_selector = MultiGreedyTokenSelectionStrategy(
                config,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported token selection strategy: {config.decode_config.token_selection_strategy}.\n"
                f"Supported strategies: {','.join([strategy.name for strategy in TokenSelectionStrategy])}"
            )

    return token_selector


def is_multi_beam(token_selection_strategy: TokenSelectionStrategy):
    match token_selection_strategy:
        case TokenSelectionStrategy.MULTI_GREEDY:
            return True

        case _:
            return False


__all__ = [
    "BaseTokenSelectionStrategy",
    "TokenSelectionStrategyConfig",
    "TokenSelectionStrategy",
    "GreedyTokenSelectionStrategy",
    "MultiGreedyTokenSelectionStrategy",
    "build_token_selector",
    "build_token_selector_config",
    "get_strategy_from_str",
    "is_ref_counted",
    "is_multi_beam",
]
