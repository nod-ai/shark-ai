from .config import SchedulerConfig
from .factory import _SchedulerEngineImpl, _create_scheduler


class SchedulerFacade:
    def __init__(self, impl: _SchedulerEngineImpl) -> None:
        self._impl = impl

    def should_execute(self, pending):
        return self._impl.should_execute(pending)

    def handle_message(self, msg):
        return self._impl.handle_message(msg)

    def reserve_workload(self, count: int, rid: str):
        return self._impl.reserve_workload(count=count, rid=rid)

    @staticmethod
    def build_scheduler(
        scheduler_config: SchedulerConfig,
    ):
        return SchedulerFacade(impl=_create_scheduler(scheduler_config))
