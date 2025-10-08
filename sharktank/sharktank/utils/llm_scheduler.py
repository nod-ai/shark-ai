import iree.runtime
import numpy
import torch


from typing import Callable, List, Optional, Tuple, TypeVar

from sharktank.utils.llm_tasks import (
    LlmTask,
    LlmTaskInput,
)


class Scheduler:
    def __init__(
        self,
        batch_size: int,
        block_seq_stride: int,
        llm_task_class: type[LlmTask],
        invocation_fn: Callable[
            [List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]],
            Tuple[numpy.ndarray, Optional[numpy.ndarray]],
        ],
    ) -> None:
        self._batch_size = batch_size
        self._pending_tasks = []
        self._block_stride = block_seq_stride
        self._llm_task_class = llm_task_class
        self._invocation_fn = invocation_fn

    def schedule_task(self, task: LlmTaskInput) -> None:
        self._pending_tasks.append(task)

    def get_next_batch(self) -> list[LlmTaskInput]:
        batch = self._pending_tasks[: self._batch_size]
        self._pending_tasks = self._pending_tasks[self._batch_size :]
        return batch

    def has_pending_tasks(self) -> bool:
        return len(self._pending_tasks) > 0

    def run(
        self,
        selection_fn: Callable[
            [numpy.ndarray, Optional[numpy.ndarray], List[int]], List[int]
        ],
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ):
        selections = []
        while self.has_pending_tasks():
            task_inputs = self.get_next_batch()

            llm_task_class = self._llm_task_class
            llm_task = llm_task_class(
                invocation_fn=self._invocation_fn,
                llm_task_inputs=task_inputs,
                batch_size=self._batch_size,
                block_stride=self._block_stride,
            )
            logits, indices = llm_task.run(*cache)

            last = selection_fn(
                logits,
                indices,
                llm_task.logit_positions,
            )

            selections.extend(last)

        return selections
