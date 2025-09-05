from typing import List
import shortfin as sf

from abc import ABC, abstractmethod

from ..messages import LlmInferenceExecRequest


class AbstractScheduler(ABC):
    @abstractmethod
    def should_execute(
        self, pending: List[LlmInferenceExecRequest]
    ) -> List[LlmInferenceExecRequest]:
        pass

    @abstractmethod
    def handle_message(self, message: sf.Message) -> bool:
        pass

    @abstractmethod
    def reserve_workload(self, *, count: int, rid: str):
        pass


class AbstractSchedulerRuntime(ABC):
    @abstractmethod
    def get_tick(self) -> int:
        pass

    @abstractmethod
    def submit_workload(self, *, count: int, rid: str):
        pass
