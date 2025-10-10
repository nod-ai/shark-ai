import bisect
import iree.runtime
import numpy
import torch

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from sharktank.utils.llm_tasks import (
    LlmTask,
    PrefillTask,
    LlmTaskInput,
)


class Scheduler(ABC):
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
        self._block_stride = block_seq_stride
        self._llm_task_class = llm_task_class
        self._invocation_fn = invocation_fn

    @abstractmethod
    def schedule_task(self, task: LlmTaskInput):
        pass

    @abstractmethod
    def _has_pending_tasks(self) -> bool:
        pass

    @abstractmethod
    def _get_next_batch(self) -> list[LlmTaskInput]:
        pass

    @abstractmethod
    def run(
        self,
        selection_fn: Callable[
            [numpy.ndarray, Optional[numpy.ndarray], List[int]], List[int]
        ],
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ) -> Dict[str, int]:
        pass


class BasicScheduler(Scheduler):
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
        super().__init__(
            batch_size=batch_size,
            block_seq_stride=block_seq_stride,
            llm_task_class=llm_task_class,
            invocation_fn=invocation_fn,
        )
        self._pending_tasks: list[LlmTaskInput] = []

    def schedule_task(self, task: LlmTaskInput) -> None:
        self._pending_tasks.append(task)

    def _get_next_batch(self) -> list[LlmTaskInput]:
        batch = self._pending_tasks[: self._batch_size]
        self._pending_tasks = self._pending_tasks[self._batch_size :]
        return batch

    def _has_pending_tasks(self) -> bool:
        return len(self._pending_tasks) > 0

    def run(
        self,
        selection_fn: Callable[
            [numpy.ndarray, Optional[numpy.ndarray], List[int]], List[int]
        ],
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ):
        selections = {}
        while self._has_pending_tasks():
            task_inputs = self._get_next_batch()

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

            for i, task in enumerate(task_inputs):
                selections[task.request_id] = last[i]

        return selections


class ChunkScheduler(Scheduler):
    def __init__(
        self,
        batch_size: int,
        block_seq_stride: int,
        llm_task_class: type[LlmTask],
        invocation_fn: Callable[
            [List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]],
            Tuple[numpy.ndarray, Optional[numpy.ndarray]],
        ],
        has_prefill_position: bool,
        chunk_block_size: int,
    ) -> None:
        assert has_prefill_position, "ChunkScheduler requires has_prefill_position=True"
        assert chunk_block_size is not None, "ChunkScheduler requires chunk_block_size"

        self._has_prefill_position = has_prefill_position
        self._chunk_block_size = chunk_block_size

        # `pending_tasks` contains chunks that cannot yet be safely invoke on
        # while `ready_tasks` contains chunks that can safely be included in a batch.
        # When the `n - 1` chunk is scheduled for invocation, the `nth` chunk
        # with the same `request_id` will be moved from `pending_tasks` to `ready_tasks`.
        self._pending_tasks: Dict[str, List[LlmTaskInput]] = {}
        self._ready_tasks: List[LlmTaskInput] = []
        super().__init__(
            batch_size=batch_size,
            block_seq_stride=block_seq_stride,
            llm_task_class=llm_task_class,
            invocation_fn=invocation_fn,
        )

    def schedule_task(self, task: LlmTaskInput):
        if task.request_id not in self._pending_tasks:
            self._pending_tasks[task.request_id] = []
            self._ready_tasks.append(task)
            return

        bisect.insort(
            self._pending_tasks[task.request_id],
            task,
            key=lambda x: x.chunk_id,
        )

    def _get_next_batch(self):
        batch = self._ready_tasks[: self._batch_size]
        self._ready_tasks = self._ready_tasks[self._batch_size :]
        for task in batch:
            if len(self._pending_tasks[task.request_id]) == 0:
                del self._pending_tasks[task.request_id]
                continue

            next_task = self._pending_tasks[task.request_id].pop(0)
            self._ready_tasks.append(next_task)

        return batch

    def _has_pending_tasks(self):
        return len(self._ready_tasks) > 0 or len(self._pending_tasks) > 0

    def run(
        self,
        selection_fn: Callable[
            [numpy.ndarray, Optional[numpy.ndarray], List[int]], List[int]
        ],
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ):
        selections = {}
        while self._has_pending_tasks():
            task_inputs = self._get_next_batch()

            llm_task = self._llm_task_class(
                invocation_fn=self._invocation_fn,
                llm_task_inputs=task_inputs,
                batch_size=self._batch_size,
                block_stride=self._block_stride,
                has_prefill_position=self._has_prefill_position,
                chunk_block_size=self._chunk_block_size,
            )
            logits, indices = llm_task.run(*cache)

            last = selection_fn(
                logits,
                indices,
                llm_task.logit_positions,
            )

            for i, task in enumerate(task_inputs):
                if len(self._pending_tasks.get(task.request_id, [])) == 0 and (
                    task not in self._ready_tasks
                ):
                    selections[task.request_id] = last[i]

        return selections
