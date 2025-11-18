import bisect
import iree.runtime
import numpy
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from amdsharktank.utils.llm_tasks import (
    LlmTask,
    PrefillTask,
    LlmTaskInput,
)

SelectionFn = Callable[[numpy.ndarray, Optional[numpy.ndarray], List[int]], List[Any]]
"""The transformation to apply on the resulting (logits, indices, positions) for each
batch."""

InvocationFn = Callable[
    [List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]],
    Tuple[numpy.ndarray, Optional[numpy.ndarray]],
]
"""The invocation function passed to a scheduler. This is the model execution function.
E.g. LLM decode, prefill, etc."""


class Scheduler(ABC):
    def __init__(
        self,
        batch_size: int,
        block_seq_stride: int,
        llm_task_class: type[LlmTask],
        invocation_fn: InvocationFn,
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
        selection_fn: SelectionFn,
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ) -> Dict[str, list[Any]]:
        """Returns a map from request ID to a list of all selections for
        tasks/chunks associated with a request.
        The tasks results for a request have the same order as scheduled."""
        pass


class BasicScheduler(Scheduler):
    def __init__(
        self,
        batch_size: int,
        block_seq_stride: int,
        llm_task_class: type[LlmTask],
        invocation_fn: InvocationFn,
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
        selection_fn: SelectionFn,
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ):
        selections: dict[str, list[Any]] = {}
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

            selection = selection_fn(
                logits,
                indices,
                llm_task.logit_positions,
            )

            for i, task in enumerate(task_inputs):
                selections[task.request_id] = [selection[i]]

        return selections


class ChunkScheduler(Scheduler):
    def __init__(
        self,
        batch_size: int,
        block_seq_stride: int,
        llm_task_class: type[LlmTask],
        invocation_fn: InvocationFn,
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
        selection_fn: SelectionFn,
        cache: List[iree.runtime.DeviceArray | torch.Tensor],
    ):
        selections: defaultdict[str, list[Any]] = defaultdict(list)
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

            selection = selection_fn(
                logits,
                indices,
                llm_task.logit_positions,
            )

            for i, task in enumerate(task_inputs):
                assert task.chunk_id == len(
                    selections[task.request_id]
                ), f"Out-of-order tasks for request {task.request_id}. Expected to be ordered by prefill chunk"
                selections[task.request_id].append(selection[i])
        return selections
