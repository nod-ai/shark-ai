from dataclasses import dataclass
from sharktank.utils.llm_task import LlmTaskInput
from typing import List, Dict


@dataclass
class SchedulerEntry:
    task: LlmTaskInput
    remaining_count: int


class Scheduler:
    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._queue: List[LlmTaskInput] = []
        self._schedule_count: Dict[str, SchedulerEntry] = {}

    def schedule_task(self, task: LlmTaskInput, count: int):
        self._queue.append(task)
        self._schedule_count[task.task_id] = SchedulerEntry(
            task=task, remaining_count=count
        )

    def get_scheduled_tasks(self) -> List[LlmTaskInput]:
        batch_size = self._batch_size
        schedule_tasks = self._queue[:batch_size]
        self._queue = self._queue[batch_size:]

        for task in schedule_tasks:
            self._schedule_count[task.task_id].remaining_count -= 1

        return schedule_tasks

    def has_pending_tasks(self) -> bool:
        return len(self._queue) > 0

    def on_task_complete(self, task_id: str, last_token: int) -> bool:
        task = self._schedule_count[task_id].task
        task.tokens.append(last_token)
        task.seq_len = len(task.tokens)
        task.start_position = task.seq_len - 1

        if self._schedule_count[task_id].remaining_count == 0:
            del self._schedule_count[task_id]
            return True

        self._queue.append(self._schedule_count[task_id].task)
        return False

    def remove_task(self, task_id: str):
        self._queue = [task for task in self._queue if task.task_id != task_id]
        if task_id in self._schedule_count:
            del self._schedule_count[task_id]
