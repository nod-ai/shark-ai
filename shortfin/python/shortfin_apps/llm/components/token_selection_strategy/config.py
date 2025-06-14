# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, fields
from typing import Callable, List, Union
from dataclasses_json import dataclass_json, Undefined
from enum import Enum, auto


from ..io_struct import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_TEMPERATURE, NOT_PROVIDED
from ..messages import LlmInferenceExecRequest


class LogitsNormalization(Enum):
    """Supported token selection strategies."""

    NONE = auto()
    SOFTMAX = auto()
    LOG_SOFTMAX = auto()

    @classmethod
    def _missing_(cls, value):
        value = value.upper()
        for member in cls:
            if member.name == value:
                return member
        raise KeyError(f"Unknown token_selection_strategy: {value}")


def get_normalization_from_str(token_selection_strategy: str) -> LogitsNormalization:
    name_to_strategy = {
        strategy.name.lower(): strategy for strategy in LogitsNormalization
    }
    strategy = token_selection_strategy.lower()
    if strategy not in name_to_strategy:
        raise KeyError(f"Unknown token_selection_strategy: {token_selection_strategy}")


class TokenSelectionStrategy(Enum):
    """Supported token selection strategies."""

    INDEPENDENT = auto()
    BEAM_SEARCH = auto()


def get_strategy_from_str(token_selection_strategy: str) -> TokenSelectionStrategy:
    name_to_strategy = {
        strategy.name.lower(): strategy for strategy in TokenSelectionStrategy
    }
    strategy = token_selection_strategy.lower()
    if strategy not in name_to_strategy:
        raise KeyError(f"Unknown token_selection_strategy: {token_selection_strategy}")

    return name_to_strategy[strategy]


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DecodeConfig:

    # Number of beams to use during generation
    num_beams: int = 1

    logits_normalization: LogitsNormalization = LogitsNormalization.NONE

    # Max number of tokens to generate in decode loop
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS

    # Flatten or stretch logits to increase variability
    temperature: float = DEFAULT_TEMPERATURE

    # Whether or not to use beam search during generation
    use_beam_search: bool = False

    # Use `top_k` sampling strategy in decode loop
    top_k: int | None = None

    # Use `top_p` sampling strategy in decode loop
    top_p: int | None = None

    def update_from_sampling_params(self, sampling_params):
        for field in fields(sampling_params):
            if getattr(sampling_params, field.name) == NOT_PROVIDED:
                continue
            if hasattr(self, field.name):
                setattr(self, field.name, getattr(sampling_params, field.name))


@dataclass
class TokenSelectionStrategyConfig:
    """Configuration for token selection strategies."""

    decode_config: DecodeConfig
    prefill_callback: Callable[[LlmInferenceExecRequest], None]
    decode_callback: Callable[[LlmInferenceExecRequest], None]
    decode_begin_callback: Callable[[int], None]
    decode_end_callback: Callable[[int], None]
    results_callback: Callable[[Union[int, List[int]]], None]
    eos_token_id: int
