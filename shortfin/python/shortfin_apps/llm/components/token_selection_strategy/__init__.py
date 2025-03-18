# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, List, Union
from .base_token_selection_strategy import (
    BaseTokenSelectionStrategy,
    TokenSelectionStrategyConfig,
    TokenSelectionStrategy,
)
from .greedy_token_selection_strategy import GreedyTokenSelectionStrategy

from ..messages import LlmInferenceExecRequest


def build_token_selector_config(
    token_selection_strategy: TokenSelectionStrategy,
    prefill_callback: Callable[[LlmInferenceExecRequest], None],
    decode_callback: Callable[[LlmInferenceExecRequest], None],
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
    match token_selection_strategy:
        case TokenSelectionStrategy.GREEDY:
            config = TokenSelectionStrategyConfig(
                token_selection_strategy,
                prefill_callback=prefill_callback,
                decode_callback=decode_callback,
                results_callback=results_callback,
                eos_token_id=eos_token_id,
                max_completion_tokens=max_completion_tokens,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported token selection strategy: {token_selection_strategy}.\n"
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
    match config.token_selection_strategy:
        case TokenSelectionStrategy.GREEDY:
            token_selector = GreedyTokenSelectionStrategy(
                config,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported token selection strategy: {config.token_selection_strategy}.\n"
                f"Supported strategies: {','.join([strategy.name for strategy in TokenSelectionStrategy])}"
            )

    return token_selector


__all__ = [
    "BaseTokenSelectionStrategy",
    "TokenSelectionStrategyConfig",
    "TokenSelectionStrategy",
    "GreedyTokenSelectionStrategy",
    "build_token_selector",
    "build_token_selector_config",
]
