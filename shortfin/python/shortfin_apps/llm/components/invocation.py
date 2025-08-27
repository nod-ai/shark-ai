import logging
import math
import traceback

import shortfin as sf
import shortfin.array as sfnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Tuple, Union

from .buffers import copy_buffers_to_host, create_argument_buffers
from .kvcache.base_attention_cache import BasePagedAttentionCache
from .device_array_cache import Allocation, DeviceArrayCache, WrappedAllocation
from .messages import LlmInferenceExecRequest, InferencePhase


logger = logging.getLogger(__name__)


@dataclass
class LlmTaskInput:
    block_count: int
    seq_stride: int
    input_tokens: List[List[int]]
    page_ids: List[List[int]]

    start_positions: Optional[List[int]] = None

    @property
    def batch_seq_len(self):
        seq_stride = self.seq_stride
        bsl = max(len(tokens) for tokens in self.input_tokens)
        return int(math.ceil(bsl / seq_stride) * seq_stride)


class LlmTaskResponder(ABC):
    def __init__(self, exec_requests):
        self._exec_requests = exec_requests

    @abstractmethod
    def set_success(
        self, logits: sfnp.device_array, indices: Optional[sfnp.device_array]
    ):
        ...

    def set_failure(self, exception):
        ...


class LlmTask:
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_inputs: LlmTaskInput,
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
    ):
        self.req_count = len(task_inputs.input_tokens)

        self._task_input = task_inputs
        self._array_cache: DeviceArrayCache = array_cache
        self._page_tables = page_tables

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
        task_inputs: LlmTaskInput,
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
    ):
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
        )

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
        task_inputs = self._task_input

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        batch_seq_len = task_inputs.batch_seq_len
        logger.debug(f"Prefill bs={batch_size}, bsl={batch_seq_len}")

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, batch_seq_len], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate(
            [batch_size, task_inputs.block_count], int_dtype
        )

        # Prepare data for argument buffers
        tokens_data = list(
            chain.from_iterable(
                _pad_list(tokens, task_inputs.batch_seq_len)
                for tokens in task_inputs.input_tokens
            )
        )

        seq_lens_data = [len(tokens) for tokens in task_inputs.input_tokens]

        seq_block_ids_data = list(
            chain.from_iterable(
                _pad_list(pages, target_length=task_inputs.block_count)
                for pages in task_inputs.page_ids
            )
        )

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, seq_block_ids],
            data=[tokens_data, seq_lens_data, seq_block_ids_data],
            defaults=[0, 1, 0],
        )

        for page_table in self._page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args


class DecodeTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        task_inputs: LlmTaskInput,
        array_cache: DeviceArrayCache,
        page_tables: List[sfnp.device_array],
    ):
        assert (
            task_inputs.start_positions is not None
        ), "`start_positions` must be defined for `Decode`."
        super().__init__(
            task_inputs=task_inputs,
            array_cache=array_cache,
            page_tables=page_tables,
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
        task_inputs = self._task_input
        block_count = task_inputs.block_count
        logger.debug("Decode bs=%d", self.req_count)

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, 1], int_dtype)
        start_positions = array_cache.allocate([batch_size], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate([batch_size, block_count], int_dtype)

        # Prepare data for argument buffers
        tokens_data = list(
            chain.from_iterable(tokens[-1:] for tokens in task_inputs.input_tokens)
        )
        seq_lens_data = [pos + 1 for pos in task_inputs.start_positions]

        seq_block_ids_data = list(
            chain.from_iterable(
                _pad_list(pages, block_count) for pages in task_inputs.page_ids
            )
        )

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, start_positions, seq_block_ids],
            data=[
                tokens_data,
                seq_lens_data,
                task_inputs.start_positions,
                seq_block_ids_data,
            ],
            defaults=[0, 1, 0, 0],
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

            self._responder.set_success(logits, indices)

        except Exception as exception:
            self._responder.set_failure(exception)
