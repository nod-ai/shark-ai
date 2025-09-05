from .abstract import AbstractSchedulerRuntime

from dataclasses import dataclass
from enum import auto, Enum


class SchedulerMode(Enum):
    STROBE = auto()


@dataclass
class SchedulerConfig:
    mode: SchedulerMode
    ideal_batch_size: int
    runtime: AbstractSchedulerRuntime
