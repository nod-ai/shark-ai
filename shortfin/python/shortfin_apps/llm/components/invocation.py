import logging
import math

import shortfin as sf
import shortfin.array as sfnp

from typing import List, Optional, Tuple, Union

from .buffers import copy_buffers_to_host, create_argument_buffers
from .device_array_cache import Allocation, DeviceArrayCache, WrappedAllocation
from .messages import LlmInferenceExecRequest


logger = logging.getLogger(__name__)


class LlmTask:
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        exec_requests: List[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
    ):
        self.exec_requests: List[LlmInferenceExecRequest] = exec_requests
        self._array_cache: DeviceArrayCache = array_cache
        self._seq_stride: int = seq_stride

    def get_args_data(
        self,
        exec_requests: List[LlmInferenceExecRequest],
        *args,
    ) -> Tuple[List[int | float] | List[List[int | float]]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Args:
            exec_requests (List[LlmInferenceExecRequest]): List of execution requests.
            *args: Additional arguments that may be needed for specific implementations.

        Returns:
            Tuple[List[int | float] | List[List[int | float]]]: A tuple containing argument data.
        """

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[list[sfnp.device_array], int]:
        """Get the arguments for the invocation.

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[list[sfnp.device_array], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError(
            "get_args must be implemented in subclasses of LlmTask"
        )

    async def post_process_logits(
        self,
        args: List[Union[Allocation, WrappedAllocation]],
        req_count: int,
        result: Tuple[sfnp.device_array, Optional[sfnp.device_array]],
        device0: sf.ScopedDevice,
    ):
        """Handle the post-processing of logits after a prefill invocation.

        Args:
            args (List[Union[Allocation, WrappedAllocation]]): The arguments used for the invocation.
            req_count (int): The number of requests in the batch.
            result (Tuple[sfnp.device_array, Optional[sfnp.device_array]]): The result from the invocation.
                - The 0th element should be the `logits`
                - The 1st element should be the `indices`, if any.
            device0 (sf.ScopedDevice): The device used for invocation.
        """
        indices = None
        logits = result[0]
        if len(result) > 1:
            indices = result[1]

        # TODO (stbaione): Move this to the `decoder`
        # publish cache pages
        for r in self.exec_requests:
            total_tokens = r.start_position + len(r.input_token_ids)
            number_of_complete_pages = total_tokens // self._seq_stride
            r.publish_allocated_pages(number_of_complete_pages)

        exec_requests = self.exec_requests
        buffers = (logits, indices)
        transfer = any([req.return_host_array for req in exec_requests])

        if not transfer:
            return buffers

        logits, indices = await copy_buffers_to_host(buffers, device0)

        [arg.release() for arg in args]

        await self.get_result(logits, indices, req_count)

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        req_count: int,
    ):
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """


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
        exec_requests: list[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
    ):
        super().__init__(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
        )

    def get_args_data(
        self,
        exec_requests: List[LlmInferenceExecRequest],
        batch_seq_len: int,
        block_count: int,
    ) -> Tuple[List[int | float]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Args:
            exec_requests (List[LlmInferenceExecRequest]): List of execution requests.
            batch_seq_len (int): The maximum sequence length for the batch.
            block_count (int): The number of blocks in the sequence.

        Returns:
            Tuple[List[int | float]: A tuple containing:
                - A list of token IDs for the invocation.
                - A list of sequence lengths.
                - A list of sequence block IDs.
        """
        token_vals = [
            input_tokens
            for req in exec_requests
            for input_tokens in (_pad_list(req.input_token_ids, batch_seq_len))
        ]

        seq_lens_vals = [len(req.input_token_ids) for req in exec_requests]

        seq_block_ids_vals = []
        for req in exec_requests:
            block_ids = req.cache_page_indices(block_count)
            # Pad the block IDs to match the block count.
            block_ids = _pad_list(
                block_ids,
                target_length=block_count,
            )
            # Extend the sequence block IDs data with padded values.
            seq_block_ids_vals.extend(block_ids)

        return token_vals, seq_lens_vals, seq_block_ids_vals

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[list[sfnp.device_array], int]:
        """Get the arguments for the prefill invocation.

        The prefill args that are created are:
            - tokens: [bs, bsl]
            - seq_lens: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[list[sfnp.device_array], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.
        """
        exec_requests = self.exec_requests
        req_count = len(exec_requests)
        seq_stride = self._seq_stride

        for r in exec_requests:
            assert r.start_position == 0

        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        bsl = max((len(r.input_token_ids)) for r in exec_requests)
        bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
        block_count = max(r.block_count for r in exec_requests)
        req_count = len(exec_requests)
        logger.debug("Prefill bs=%d, bsl=%d", batch_size, bsl)

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, bsl], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate([batch_size, block_count], int_dtype)

        # Populate data for args.
        arg_data = self.get_args_data(
            exec_requests=exec_requests,
            batch_seq_len=bsl,
            block_count=block_count,
        )

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, seq_block_ids],
            data=arg_data,
            defaults=[0, 1, 0],
        )

        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args, req_count

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        req_count: int,
    ):
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """
        for i in range(req_count):
            req = self.exec_requests[i]
            sl = len(req.input_token_ids) - 1

            if logits.shape[1] == 1:
                logits_item = logits.view(i)
            else:
                logits_item = logits.view(i, sl)

            index_item = None
            if indices is not None:
                if indices.shape[1] == 1:
                    index_item = indices.view(i)
                else:
                    index_item = indices.view(i, sl)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self.exec_requests:
            req.done.set_success()


class DecodeTask(LlmTask):
    """Handles the transfer and preparation of data for VMFB invocation."""

    def __init__(
        self,
        exec_requests: list[LlmInferenceExecRequest],
        array_cache: DeviceArrayCache,
        seq_stride: int,
    ):
        super().__init__(
            exec_requests=exec_requests,
            array_cache=array_cache,
            seq_stride=seq_stride,
        )

    def get_args_data(
        self,
        exec_requests: List[LlmInferenceExecRequest],
        block_count: int,
    ) -> Tuple[List[int | float] | List[List[int | float]]]:
        """Get the invocation data for the given requests.

        Prepare the data that will be used to create the argument_buffers
        for the invocation.

        Args:
            exec_requests (List[LlmInferenceExecRequest]): List of execution requests.
            block_count (int): The number of blocks in the sequence.

        Returns:
            Tuple[List[int | float] | List[List[int | float]]]: A tuple containing:
                - A list of token IDs for the invocation.
                - A list of sequence lengths.
                - A list of start positions.
                - A list of sequence block IDs.
        """
        token_data = [
            input_tokens
            for req in exec_requests
            for input_tokens in (req.input_token_ids[-1:])
        ]
        seq_lens_data = [req.start_position + 1 for req in exec_requests]
        start_positions_data = [req.start_position for req in exec_requests]

        seq_block_ids_data = []
        for req in exec_requests:
            block_ids = req.cache_page_indices(block_count)
            # Pad the block IDs to match the block count.
            padded = _pad_list(
                block_ids,
                target_length=block_count,
            )
            # Extend the sequence block IDs data with padded values.
            seq_block_ids_data.extend(padded)

        return (
            token_data,
            seq_lens_data,
            start_positions_data,
            seq_block_ids_data,
        )

    async def get_args(
        self,
        page_tables: sfnp.device_array,
        batch_size: int,
    ) -> Tuple[List[Union[Allocation, WrappedAllocation]], int]:
        """Get the arguments for the decode invocation.

        The decode args that are created are:
            - tokens: [bs, 1]
            - seq_lens: [bs]
            - start_positions: [bs]
            - seq_block_ids: [bs, blocks]
            - cache_slabs: ...

        Args:
            page_tables (sfnp.device_array): Page tables in KVCache.
            batch_size (int): Size of the `exec_requests` batch.

        Returns:
            Tuple[List[Union[Allocation, WrappedAllocation]], int]: A tuple containing:
                - A list of arguments for the invocation.
                - The number of requests in the batch.
        """
        # Compute block sequence length as maximum sequence length, rounded
        # up to the seq_stride.
        exec_requests = self.exec_requests
        block_count = max(r.block_count for r in exec_requests)
        req_count = len(exec_requests)
        logger.debug("Decode bs=%d", req_count)

        array_cache = self._array_cache
        int_dtype = sfnp.int64

        # Acquire buffers for the arguments.
        tokens = array_cache.allocate([batch_size, 1], int_dtype)
        start_positions = array_cache.allocate([batch_size], int_dtype)
        seq_lens = array_cache.allocate([batch_size], int_dtype)
        seq_block_ids = array_cache.allocate([batch_size, block_count], int_dtype)

        # Populate data for args.
        args_data = self.get_args_data(
            exec_requests=exec_requests,
            block_count=block_count,
        )

        args = create_argument_buffers(
            buffers=[tokens, seq_lens, start_positions, seq_block_ids],
            data=args_data,
            defaults=[0, 1, 0, 0],
        )

        for page_table in page_tables:
            args.append(WrappedAllocation(sfnp.disable_barrier(page_table)))

        return args, req_count

    async def get_result(
        self,
        logits: sfnp.device_array,
        indices: Optional[sfnp.device_array],
        req_count: int,
    ):
        """Get the results after a prefill invocation.

        Args:
            logits (sfnp.device_array): The logits output from prefill.
            indices (Optional[sfnp.device_array]): The indices output from prefill, if any.
            req_count (int): The number of requests in the batch.
        """
        for i in range(req_count):
            req = self.exec_requests[i]
            logits_item = logits.view(i, 0)

            index_item = None
            if indices is not None:
                index_item = indices.view(i, 0)

            req.result_logits = logits_item
            req.result_indices = index_item

        for req in self.exec_requests:
            req.done.set_success()


class LlmInvoker(sf.Process):
    """Executes the invocation of LLM for a batch of requests."""

    def __init__(
        self,
        name: str,
        fiber: sf.Fiber,
        array_cache: DeviceArrayCache,
        llm_task: LlmTask,
        functions: dict[int, sf.ProgramFunction],
        seq_stride: int,
        page_tables,
        program_isolation: sf.ProgramIsolation,
    ):
        super().__init__(fiber=fiber)
        self.name = name
        self.seq_stride = seq_stride
        self.page_tables = page_tables
        self.functions = functions
        self.program_isolation = program_isolation

        self.device0 = fiber.device(0)
        self.array_cache = array_cache
        self.llm_task = llm_task

    async def run(self):
        """Invoke `prefill` or `decode` function, with IREE, on a batch of requests.

        Raises:
            RuntimeError: No available entry point for given batch size.
        """
        try:
            req_bs = len(self.llm_task.exec_requests)

            # Select an entrypoint for the batch.
            entrypoints = self.functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            args, req_count = await self.llm_task.get_args(self.page_tables, bs)

            # Invoke VMFB. Logits are of shape [bs, bsl, d].
            args_device = [arg.device for arg in args]
            result = await fn(*args_device, fiber=self.fiber)

            await self.llm_task.post_process_logits(
                args,
                req_count,
                result,
                self.device0,
            )

        except Exception:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.llm_task.exec_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()
