from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Union

from ..messages import LlmInferenceExecRequest


@dataclass
class DecodeStrategyConfig:
    batcher_callback: Callable[[LlmInferenceExecRequest], None]
    streaming_callback: Callable[[Union[int, List[int]]], None]
    eos_token_id: int
    max_completion_tokens: int


class DecodeStrategy(ABC):
    """Abstract class for implementing decode strategies."""

    @property
    @abstractmethod
    def decode_strategy_config(self) -> DecodeStrategyConfig:
        pass

    @abstractmethod
    async def decode(self) -> List[int]:
        pass
