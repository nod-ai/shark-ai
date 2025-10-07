import dataclasses
import iree.runtime
import numpy
import torch

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple


def fill_page_table(bs: int, count: int, page_ids: list[list[int]]) -> numpy.ndarray:
    pages = numpy.zeros((bs, count), dtype=numpy.int64)

    for i, ids in enumerate(page_ids):
        pages[i, : len(ids)] = ids[:count]

    return pages


@dataclasses.dataclass
class LlmTaskInput:
    task_id: str
    tokens: List[int]
    seq_len: int
    pages: List[int]

    start_position: Optional[int] = None


class LlmTask(ABC):
    def __init__(
        self,
        invocation_fn: Callable[
            [List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]],
            Tuple[numpy.ndarray, Optional[numpy.ndarray]],
        ],
        llm_task_inputs: List[LlmTaskInput],
        batch_size: int,
        block_stride: int,
    ):
        self._invocation_fn = invocation_fn
        self._task_inputs: List[LlmTaskInput] = llm_task_inputs
        self._batch_size = batch_size
        self._block_stride = block_stride

    @abstractmethod
    def _prepare_args(
        self, task_inputs: List[LlmTaskInput], *cache
    ) -> List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]:
        pass

    @abstractmethod
    def _process_results(
        self, results
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        pass

    @property
    @abstractmethod
    def logit_positions(self) -> List[int]:
        pass

    def run(
        self, *cache_state: List[iree.runtime.DeviceArray | torch.Tensor]
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        task_inputs = self._task_inputs

        args = self._prepare_args(task_inputs, *cache_state)
        results = self._invocation_fn(*args)
        logits, indices = self._process_results(results)
        return logits, indices


class PrefillTask(LlmTask):
    @property
    def logit_positions(self) -> List[int]:
        return [task_input.seq_len - 1 for task_input in self._task_inputs]

    def _prepare_args(
        self,
        task_inputs: List[LlmTaskInput],
        *cache: iree.runtime.DeviceArray | torch.Tensor,
    ) -> List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]:
        block_stride = self._block_stride
        bs = self._batch_size

        tokens = [task_input.tokens for task_input in task_inputs]
        page_ids = [task_input.pages for task_input in task_inputs]

        max_len = max(len(input_tokens) for input_tokens in tokens)
        blocks = int(numpy.ceil(max_len / block_stride))
        blocked_len = blocks * block_stride

        tokens_ = numpy.zeros((bs, blocked_len), dtype=numpy.int64)
        lens_ = numpy.ones((bs,), dtype=numpy.int64)

        for i, input_tokens in enumerate(tokens):
            tokens_[i, : len(input_tokens)] = input_tokens
            lens_[i] = len(input_tokens)

        pages_ = fill_page_table(bs, blocks, page_ids)
        args = [
            tokens_,
            lens_,
            pages_,
        ] + list(cache)
        return args

    def _process_results(
        self, results
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        if isinstance(results, tuple):
            logits, indices = results
            logits = numpy.asarray(logits)
            indices = numpy.asarray(indices)
        else:
            logits = numpy.asarray(results)
            indices = None
        return logits, indices


class DecodeTask(LlmTask):
    def __init__(
        self, *llm_task_args, decode_topk_logits: int | None = 8, **llm_task_kwargs
    ):
        super().__init__(*llm_task_args, **llm_task_kwargs)
        self._decode_topk_logits = decode_topk_logits

    @property
    def logit_positions(self) -> List[int]:
        return [0 for _ in self._task_inputs]

    def _prepare_args(
        self,
        task_inputs: List[LlmTaskInput],
        *cache: iree.runtime.DeviceArray | torch.Tensor,
    ) -> List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]:
        assert all(
            task_input.start_position is not None for task_input in task_inputs
        ), "`start_positions` is a required argument for `decode`"

        block_stride = self._block_stride
        decode_bs = self._batch_size
        bs = len(task_inputs)

        tokens = [task_input.tokens[-1] for task_input in task_inputs]
        page_ids = [task_input.pages for task_input in task_inputs]
        start_positions = [task_input.start_position for task_input in task_inputs]

        max_len = max(start_positions) + 1
        blocks = int(numpy.ceil(max_len / block_stride))

        tokens_ = numpy.zeros((decode_bs, 1), dtype=numpy.int64)
        lens_ = numpy.ones((decode_bs,), dtype=numpy.int64)
        pos_ = numpy.ones((decode_bs,), dtype=numpy.int64)

        for i in range(bs):
            tokens_[i, 0] = tokens[i]
            lens_[i] = start_positions[i] + 1
            pos_[i] = start_positions[i]

        pages_ = fill_page_table(decode_bs, blocks, page_ids)

        args = [
            tokens_,
            lens_,
            pos_,
            pages_,
        ] + list(cache)
        return args

    def _process_results(
        self, results
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        if isinstance(results, tuple):
            logits, indices = results
        else:
            if self._decode_topk_logits is None:
                logits = results
                indices = torch.broadcast_to(
                    torch.arange(results.shape[-1]), logits.shape
                )
            else:
                logits = torch.asarray(numpy.asarray(results))
                logits, indices = torch.topk(logits, self._decode_topk_logits)

        logits = numpy.asarray(logits)
        indices = numpy.asarray(indices)
        return logits, indices
