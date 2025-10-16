import iree.runtime
import numpy
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


def fill_page_table(bs: int, count: int, page_ids: list[list[int]]) -> numpy.ndarray:
    pages = numpy.zeros((bs, count), dtype=numpy.int64)

    for i, ids in enumerate(page_ids):
        pages[i, : len(ids)] = ids[:count]

    return pages


@dataclass
class LlmRequest:
    request_id: str
    tokens: List[int]
    pages: List[int]


@dataclass
class LlmTaskInput:
    request_id: str
    chunk_id: int
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
    def _process_results(self, results) -> Tuple[numpy.ndarray, numpy.ndarray]:
        pass

    @property
    @abstractmethod
    def logit_positions(self) -> List[int]:
        pass

    def _get_blocked_token_len(self, task_inputs: List[LlmTaskInput]) -> int:
        max_bsl = 0
        seq_stride = self._block_stride
        for task_input in task_inputs:
            token_len = len(task_input.tokens)
            max_bsl = max(
                max_bsl, int(int(numpy.ceil(token_len / seq_stride)) * seq_stride)
            )

        return max_bsl

    def run(
        self, *cache_state: List[iree.runtime.DeviceArray | torch.Tensor]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        task_inputs = self._task_inputs

        args = self._prepare_args(task_inputs, *cache_state)
        results = self._invocation_fn(*args)
        logits, indices = self._process_results(results)
        return logits, indices


class PrefillTask(LlmTask):
    def __init__(
        self,
        invocation_fn: Callable[
            [List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]],
            Tuple[numpy.ndarray, Optional[numpy.ndarray]],
        ],
        llm_task_inputs: List[LlmTaskInput],
        batch_size: int,
        block_stride: int,
        has_prefill_position: bool = False,
        chunk_block_size: int | None = None,
    ):
        self._has_prefill_position = has_prefill_position
        self._chunk_block_size = chunk_block_size
        super().__init__(invocation_fn, llm_task_inputs, batch_size, block_stride)

    @property
    def logit_positions(self) -> List[int]:
        return [len(task_input.tokens) - 1 for task_input in self._task_inputs]

    def _get_sequence_block_count(
        self, batch_seq_len: int, task_inputs: List[LlmTaskInput]
    ) -> int:
        if self._chunk_block_size is None:
            return batch_seq_len // self._block_stride

        assert all(task_input.start_position is not None for task_input in task_inputs)
        block_stride = self._block_stride
        max_start_position = max(
            task_input.start_position for task_input in task_inputs
        )

        # Number of blocks we're writing to
        write_block_span = batch_seq_len // block_stride
        # Calculate block offset based on the max start position
        max_chunk_start = max_start_position // block_stride

        # Prevent overflow in write page ids
        block_count = max_chunk_start + write_block_span
        return block_count

    def _prepare_args(
        self,
        task_inputs: List[LlmTaskInput],
        *cache: iree.runtime.DeviceArray | torch.Tensor,
    ) -> List[numpy.ndarray | iree.runtime.DeviceArray | torch.Tensor]:
        bs = self._batch_size

        tokens = [task_input.tokens for task_input in task_inputs]
        page_ids = [task_input.pages for task_input in task_inputs]

        blocked_token_len = self._get_blocked_token_len(task_inputs)
        sequence_block_count = self._get_sequence_block_count(
            blocked_token_len, task_inputs
        )

        tokens_ = numpy.zeros((bs, blocked_token_len), dtype=numpy.int64)
        lens_ = numpy.ones((bs,), dtype=numpy.int64)

        for i, input_tokens in enumerate(tokens):
            tokens_[i, : len(input_tokens)] = input_tokens
            lens_[i] = task_inputs[i].seq_len

        pages_ = fill_page_table(bs, sequence_block_count, page_ids)

        args = [tokens_]
        if self._has_prefill_position:
            pos_ = numpy.zeros((bs,), dtype=numpy.int64)
            for i, task_input in enumerate(task_inputs):
                pos_[i] = task_input.start_position
            args.append(pos_)

        args.append(lens_)
        args.append(pages_)
        args += list(cache)
        return args

    def _process_results(self, results) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if isinstance(results, tuple):
            logits, indices = results
            logits = numpy.asarray(logits)
            indices = numpy.asarray(indices)
        else:
            logits = numpy.asarray(results)
            # Do not return None so that downstream code does not have to deal with
            # special cases.
            indices = (
                torch.broadcast_to(torch.arange(logits.shape[-1]), logits.shape)
                .cpu()
                .numpy()
            )
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
