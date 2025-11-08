import logging
import math

import shortfin as sf
import shortfin.array as sfnp

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Optional, Tuple, Union

from .buffers import copy_buffers_to_host, create_argument_buffers
from .device_array_cache import Allocation, DeviceArrayCache, WrappedAllocation
from .messages import LlmInferenceExecRequest


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmTaskInput:
    rid: str
    instance_id: str
    block_count: int
    seq_len: int
    input_tokens: Tuple[int, ...] = field(default_factory=tuple)
    page_ids: Tuple[int, ...] = field(default_factory=tuple)
    start_position: Optional[int] = None


class LlmTaskResponder(ABC):
    def __init__(self):
        self._exec_requests: dict[str, LlmInferenceExecRequest] = {}

    @abstractmethod
    def set_success(
        self,
        llm_task: "LlmTask",
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
    ):
        ...

    @abstractmethod
    def set_failure(self, llm_task: "LlmTask"):
        ...

    def add_request(self, exec_request: LlmInferenceExecRequest):
        self._exec_requests[exec_request.instance_id] = exec_request

    def _remove_request(self, instance_id: str):
        if instance_id in self._exec_requests:
            del self._exec_requests[instance_id]

    def _get_requests_from_task(
        self, llm_task: "LlmTask"
    ) -> List[LlmInferenceExecRequest]:
        return [
            self._exec_requests[task_input.instance_id]
            for task_input in llm_task._task_inputs
        ]


class LlmTask:
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_inputs: List[LlmTaskInput],
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
        seq_stride: int,
    ):
        self.req_count = len(task_inputs)

        self._task_inputs = task_inputs
        self._array_cache: DeviceArrayCache = array_cache
        self._page_tables = page_tables
        self._seq_stride = seq_stride

    @property
    def task_inputs(self):
        return self._task_inputs

    def _get_batch_seq_len(self, task_inputs: List[LlmTaskInput]) -> int:
        max_bsl = 0
        seq_stride = self._seq_stride
        for task_input in task_inputs:
            bsl = len(task_input.input_tokens)
            max_bsl = max(max_bsl, int(math.ceil(bsl / seq_stride) * seq_stride))

        return max_bsl

    async def prepare_args(
        self,
        batch_size: int,
    ) -> List[sfnp.device_array]:
        """Prepare the arguments for invocation.

        Args:
            batch_size (int): Batch size of the invocation function.

        Returns:
            List[sfnp.device_array]: A list of arguments for the invocation.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError(
            "get_args must be implemented in subclasses of LlmTask"
        )

    async def process_results(
        self,
        args: List[Union[Allocation, WrappedAllocation]],
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        device0: sf.ScopedDevice,
    ) -> Tuple[sfnp.device_array, Optional[sfnp.device_array]]:
        """Process the results of the invocation.

        Args:
            args (List[Union[Allocation, WrappedAllocation]]): Args used for invocation.
            logits (sfnp.device_array): Logits from invocation.
            indices (Optional[sfnp.device_array]): Indices from invocation.
            device0 (sf.ScopedDevice): Device used for invocation.

        Returns:
            Tuple[sfnp.device_array, Optional[sfnp.device_array]]:
                - First item is logits
                - Seconds items is optional indices
        """
        buffers = (logits, indices)
        logits, indices = await copy_buffers_to_host(buffers, device0)

        # Release arg allocations
        [arg.release() for arg in args]

        return logits, indices


def _pad_list(
    data: List[int | float],
    target_length: int,
) -> List[int | float]:
    """Pad a list to a target length with a specified value."""
    return data + [0] * max(0, target_length - len(data))


class PrefillTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_inputs: List[LlmTaskInput],
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
        seq_stride: int,
        has_prefill_position: bool,
        chunk_block_size: Optional[int] = None,
    ):
        self._has_prefill_position = has_prefill_position
        self._chunk_block_size = chunk_block_size
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
            seq_stride=seq_stride,
        )

    def _get_block_count(
        self, batch_seq_len: int, task_inputs: List[LlmTaskInput]
    ) -> int:
        if not self._has_prefill_position:
            return max(task_input.block_count for task_input in task_inputs)

        seq_stride = self._seq_stride
        max_start_position = max(
            task_input.start_position for task_input in task_inputs
        )
        # Number of blocks we're writing to
        write_block_span = batch_seq_len // seq_stride
        # Calculate block offset based on the maximum start position
        max_block_start = max_start_position // seq_stride
        # Prevent overflow in write page ids
        block_count = max_block_start + write_block_span
        return block_count

    async def prepare_args(
        self,
        batch_size: int,
    ) -> List[sfnp.device_array]:
        """Get the arguments for the prefill invocation.

        The prefill args that are created are:
            - tokens: [bs, bsl]
            - seq_lens: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

        Args:
            batch_size (int): Size of the invocation function batch.

        Returns:
            List[sfnp.device_array]: A list of arguments for the invocation.
        """
        task_inputs = self._task_inputs

        tokens = [list(task_input.input_tokens) for task_input in task_inputs]
        page_ids = [list(task_input.page_ids) for task_input in task_inputs]

        batch_seq_len = self._get_batch_seq_len(task_inputs)
        block_count = self._get_block_count(batch_seq_len, task_inputs)

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        logger.debug(f"Prefill bs={batch_size}, bsl={batch_seq_len}")

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        try:
            # Acquire buffers for the arguments.
            tokens_allocation = array_cache.allocate(
                [batch_size, batch_seq_len], int_dtype
            )
            seq_lens_allocation = array_cache.allocate([batch_size], int_dtype)
            seq_block_ids_allocation = array_cache.allocate(
                [batch_size, block_count], int_dtype
            )
        except Exception as e:
            error_msg = f"Device buffer allocation failed for prefill: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Prepare data for argument buffers
        tokens_data = list(
            chain.from_iterable(_pad_list(t, batch_seq_len) for t in tokens)
        )

        seq_lens_data = [task_input.seq_len for task_input in task_inputs]

        seq_block_ids_data = list(
            chain.from_iterable(
                _pad_list(pages, target_length=block_count) for pages in page_ids
            )
        )

        buffers = [tokens_allocation]
        data = [tokens_data]
        defaults = [0]

        if self._has_prefill_position:
            assert all(
                task.start_position is not None for task in task_inputs
            ), "`start_positions` must be defined for `Prefill` when `has_prefill_position` is True."
            start_positions = [task.start_position for task in task_inputs]
            try:
                start_positions_allocation = array_cache.allocate(
                    [batch_size], int_dtype
                )
            except Exception as e:
                error_msg = "Failed to allocate device buffer: start_positions"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            buffers.append(start_positions_allocation)
            data.append(start_positions)
            defaults.append(0)

        buffers.extend([seq_lens_allocation, seq_block_ids_allocation])
        data.extend([seq_lens_data, seq_block_ids_data])
        defaults.extend([0, 0])

        args = create_argument_buffers(
            buffers=buffers,
            data=data,
            defaults=defaults,
        )

        for page_table in self._page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args


class DecodeTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_inputs: List[LlmTaskInput],
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
        seq_stride: int,
    ):
        assert all(
            task_input.start_position is not None for task_input in task_inputs
        ), "`start_positions` must be defined for `Decode`."
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
            seq_stride=seq_stride,
        )

    async def prepare_args(
        self,
        batch_size: int,
    ) -> List[sfnp.device_array]:
        """Get the arguments for the decode invocation.

        The decode args that are created are:
            - tokens: [bs, 1]
            - seq_lens: [bs]
            - start_positions: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

        Args:
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            List[sfnp.device_array]: A list of arguments for the invocation.
        """
        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        task_inputs = self._task_inputs

        tokens = [list(task_input.input_tokens) for task_input in task_inputs]
        start_positions = [task_input.start_position for task_input in task_inputs]
        page_ids = [list(task_input.page_ids) for task_input in task_inputs]

        block_count = max(task_input.block_count for task_input in task_inputs)
        logger.debug("Decode bs=%d", self.req_count)

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        try:
            # Acquire buffers for the arguments.
            tokens_allocation = array_cache.allocate([batch_size, 1], int_dtype)
            start_positions_allocation = array_cache.allocate([batch_size], int_dtype)
            seq_lens_allocation = array_cache.allocate([batch_size], int_dtype)
            seq_block_ids_allocation = array_cache.allocate(
                [batch_size, block_count], int_dtype
            )
        except Exception as e:
            error_msg = f"Device buffer allocation failed for decode: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Prepare data for argument buffers
        tokens_data = list(chain.from_iterable(t[-1:] for t in tokens))

        seq_lens_data = [task_input.seq_len for task_input in task_inputs]

        seq_block_ids_data = list(
            chain.from_iterable(_pad_list(pages, block_count) for pages in page_ids)
        )

        args = create_argument_buffers(
            buffers=[
                tokens_allocation,
                seq_lens_allocation,
                start_positions_allocation,
                seq_block_ids_allocation,
            ],
            data=[
                tokens_data,
                seq_lens_data,
                start_positions,
                seq_block_ids_data,
            ],
            defaults=[0, 0, 0, 0],
        )

        for page_table in self._page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args


class ExtendAttentionPrefillTask(PrefillTask):
    """Prefill task that supports extend-attention with varying sequence lengths.

    This task is designed for extend-attention batching where requests can have
    different sequence lengths within a batch. It properly handles offset prefill
    by using the parent class's methods for calculating batch sequence length and
    block counts.
    """

    def __init__(
        self,
        task_inputs: List[LlmTaskInput],
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
        has_prefill_position: bool,
        block_seq_stride: int,
    ):
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
            seq_stride=block_seq_stride,
            has_prefill_position=has_prefill_position,
            chunk_block_size=None,
        )

    async def prepare_args(self, batch_size: int) -> List[sfnp.device_array]:
        """Prepare arguments for extend-attention prefill.

        Each request's tokens are divided into pages of
        block_seq_stride size. Each page can track its own history, allowing
        efficient batching of variable-length sequences.
        """
        task_inputs = self._task_inputs

        # Prepare the batch with page-aligned tokens
        tokens = []
        seq_lens = []
        page_ids = []
        start_positions = []

        for task_input in task_inputs:
            # Each task's tokens are organized by pages
            task_tokens = list(task_input.input_tokens)

            # Pad each sequence to page boundaries
            tokens.append(task_tokens)
            seq_lens.append(task_input.seq_len)
            page_ids.append(list(task_input.page_ids))
            if self._has_prefill_position:
                start_positions.append(task_input.start_position)

        batch_seq_len = self._get_batch_seq_len(task_inputs)
        max_blocks = self._get_block_count(batch_seq_len, task_inputs)

        logger.debug(
            f"ExtendAttention Prefill bs={batch_size}, "
            f"batch_seq_len={batch_seq_len}, "
            f"max_blocks={max_blocks}, seq_stride={self._seq_stride}"
        )

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Allocate buffers
        tokens_allocation = array_cache.allocate([batch_size, batch_seq_len], int_dtype)
        seq_lens_allocation = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids_allocation = array_cache.allocate(
            [batch_size, max_blocks], int_dtype
        )

        tokens_data = list(
            chain.from_iterable(_pad_list(t, batch_seq_len) for t in tokens)
        )

        seq_block_ids_data = list(
            chain.from_iterable(
                _pad_list(pages, target_length=max_blocks) for pages in page_ids
            )
        )

        buffers = [tokens_allocation]
        data = [tokens_data]
        defaults = [0]

        if self._has_prefill_position:
            start_positions_allocation = array_cache.allocate([batch_size], int_dtype)
            buffers.append(start_positions_allocation)
            data.append(start_positions)
            defaults.append(0)

        buffers.extend([seq_lens_allocation, seq_block_ids_allocation])
        data.extend([seq_lens, seq_block_ids_data])
        defaults.extend([0, 0])

        logger.info(
            f"ExtendAttention Prefill: batch_size={batch_size}, actual_requests={len(task_inputs)}, seq_lens={seq_lens}, defaults={defaults}"
        )

        args = create_argument_buffers(
            buffers=buffers,
            data=data,
            defaults=defaults,
        )

        for page_table in self._page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args


class LlmInvocationProcess(sf.Process):
    """Executes the invocation of LLM for a batch of requests."""

    def __init__(
        self,
        name: str,
        fiber: sf.Fiber,
        llm_task: LlmTask,
        functions: dict[int, sf.ProgramFunction],
        program_isolation: sf.ProgramIsolation,
        responder: LlmTaskResponder,
    ):
        super().__init__(fiber=fiber)
        self._name = name
        self._functions = functions
        self._program_isolation = program_isolation

        self._device0 = fiber.device(0)
        self._llm_task = llm_task
        self._responder = responder

    async def run(self):
        """Invoke `prefill` or `decode` function, with IREE, on a batch of requests.

        Raises:
            RuntimeError: No available entry point for given batch size.
        """
        try:
            req_count = self._llm_task.req_count

            # Select an entrypoint for the batch.
            entrypoints = self._functions
            for bs, fn in entrypoints.items():
                if bs >= req_count:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_count}")

            args = await self._llm_task.prepare_args(bs)
            args_device = [arg.device for arg in args]

            # Invoke VMFB. Logits are of shape [bs, bsl, d].
            results = await fn(*args_device, fiber=self.fiber)

            indices = None
            logits = results[0]
            if len(results) > 1:
                indices = results[1]

            logits, indices = await self._llm_task.process_results(
                args,
                logits,
                indices,
                self._device0,
            )

            self._responder.set_success(self._llm_task, logits, indices)

        except Exception:
            self._responder.set_failure(self._llm_task)
