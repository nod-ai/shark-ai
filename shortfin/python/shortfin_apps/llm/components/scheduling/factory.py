import shortfin as sf

from typing import List

from .abstract import AbstractScheduler
from .config import SchedulerConfig, SchedulerMode
from .modes.strobe import StrobeScheduler
from ..messages import LlmInferenceExecRequest


class _SchedulerEngineImpl:
    def __init__(self, scheduler: AbstractScheduler) -> None:
        self.scheduler = scheduler

    def should_execute(
        self, pending: List[LlmInferenceExecRequest]
    ) -> List[LlmInferenceExecRequest]:
        return self.scheduler.should_execute(pending)

    def handle_message(self, message: sf.Message) -> bool:
        return self.scheduler.handle_message(message)

    def reserve_workload(self, *, count: int, rid: str):
        self.scheduler.reserve_workload(count=count, rid=rid)


def _create_scheduler(config: SchedulerConfig) -> _SchedulerEngineImpl:
    if config.mode == SchedulerMode.STROBE:
        return _SchedulerEngineImpl(
            scheduler=StrobeScheduler(config=config),
        )

    raise ValueError(f"Unsupported scheduling mode: {config.mode}")
