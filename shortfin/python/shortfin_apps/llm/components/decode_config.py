# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass, fields
from dataclasses_json import dataclass_json, Undefined
from enum import Enum, auto


from .io_struct import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_TEMPERATURE, NOT_PROVIDED


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


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(kw_only=True)
class DecodeConfig:
    eos_token_id: int = 0

    # Number of beams to use during generation
    num_beams: int = 1

    logits_normalization: LogitsNormalization = LogitsNormalization.NONE

    # Max number of tokens to generate in decode loop
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS

    # Flatten or stretch logits to increase variability
    temperature: float = DEFAULT_TEMPERATURE

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

    def copy(self) -> "DecodeConfig":
        """Create a shallow copy of the DecodeConfig instance."""
        return DecodeConfig(
            eos_token_id=self.eos_token_id,
            num_beams=self.num_beams,
            logits_normalization=self.logits_normalization,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
