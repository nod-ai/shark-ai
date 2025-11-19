"""
llm_utils.py
============

This module provides utility classes and functions for working with Large Language Models (LLMs) in the amdsharktank framework.
It includes abstractions for model instances, integration with PyTorch and IREE backends, and helpers for configuration, batching, decoding, and evaluation.

Key functionalities:
- `LlmInstance`, `TorchInstance`, `IreeInstance`: Abstractions for managing LLMs in both eager (PyTorch) and compiled (IREE) modes.
- `llama_config_page_size`, `server_config_page_size`: Helpers to determine the page size for Llama model configurations and server configs.
- `LlmBatch`, `LlmDecoder`, `LlmBencher`, `LlmPerplexityEval`: Utilities for batching, decoding, benchmarking, and evaluating LLMs.
- Used by both test suites and command-line tools (see `toy_llama_test.py`, `run_llm_vmfb.py`) to provide a unified interface for LLM inference and evaluation.

Typical usage:
- In tests, to instantiate and evaluate LLMs for correctness and performance.
- In tools, to wrap IREE-compiled models for inference with custom configurations.

This module is not intended to be run directly, but is imported by other components in the amdsharktank codebase.
"""

import dataclasses
import iree.runtime
import itertools
import math
import numpy
import pathlib
import time
import torch

from collections.abc import Sequence
from copy import deepcopy
from datasets import load_dataset
from iree.runtime import ParameterIndex
from amdsharktank import ops
from amdsharktank.layers.configs.llm_configs import LlamaModelConfig
from amdsharktank.models.llm.config import ServiceConfig
from amdsharktank.models.llm import PagedLlmModelV1
from amdsharktank.types import Dataset, Theta
from amdsharktank.types.tensors import unbox_tensor
from amdsharktank.utils.attention import *
from amdsharktank.utils.llm_scheduler import (
    ChunkScheduler,
    BasicScheduler,
    SelectionFn,
    Scheduler,
)
from amdsharktank.utils.llm_tasks import (
    DecodeTask,
    LlmTaskInput,
    LlmRequest,
    PrefillTask,
)
from amdsharktank.utils.math import ceildiv
from typing import Callable, List, Optional

np_dtype_to_torch_dtype = {
    # This torch-to-torch map is an abuse to circumvent that numpy does not have bf16.
    torch.bfloat16: torch.bfloat16,
    torch.float8_e4m3fn: torch.float8_e4m3fn,
    torch.float8_e4m3fnuz: torch.float8_e4m3fnuz,
    numpy.float16: torch.float16,
    numpy.float32: torch.float32,
}

np_dtype_to_hal_dtype = {
    numpy.float16: iree.runtime.HalElementType.FLOAT_16,
    numpy.float32: iree.runtime.HalElementType.FLOAT_32,
    torch.float8_e4m3fn: iree.runtime.HalElementType.FLOAT_8_E4M3_FN,
    torch.float8_e4m3fnuz: iree.runtime.HalElementType.FLOAT_8_E4M3_FNUZ,
    torch.bfloat16: iree.runtime.HalElementType.BFLOAT_16,
}

dtype_string_to_type = {
    "bfloat16": torch.bfloat16,
    "float16": numpy.float16,
    "float32": numpy.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "int64": numpy.int64,
}


def llama_config_page_sizes(config: LlamaModelConfig) -> list[int]:
    return [
        config.hp.attention_head_count_kv
        * config.hp.attn_head_dim
        * num_blocks
        * config.block_seq_stride
        * 2
        for num_blocks in config.parallelism_config.num_blocks_per_pipeline
    ]


def minimum_required_kv_cache_page_count_for_batch(
    tokens: list[list[int]], config: LlamaModelConfig, decode_steps: int = 0
) -> int:
    """Compute the minimum number pages required to run the given tokens in 1 batch."""
    max_seq_len = max(len(t) for t in tokens) + decode_steps
    pages_per_seq = ceildiv(max_seq_len, config.block_seq_stride)
    batch_size = len(tokens)

    res = batch_size * pages_per_seq

    # A possible bug with IREE execution causes
    # tests/models/llama/toy_llama_test.py::TestToyLlamaIree::testDecodePerplexity
    # to return inf logits if we start the page indices from 0.
    # See https://github.com/nod-ai/amdshark-ai/issues/2355
    res += 1

    return res


def server_config_page_size(config: ServiceConfig) -> list[int]:
    elements_per_device = config.paged_kv_cache.paged_kv_block_size_elements_per_device
    if elements_per_device:
        return elements_per_device

    print(
        "WARNING: server_config_page_size is deprecated and will be removed in a future version. Use paged_kv_block_size_elements_per_device instead."
    )
    page_kv_cache = config.paged_kv_cache
    attn_head_dim = config.attn_head_dim
    attn_head_count = page_kv_cache.attention_head_count_kv
    block_seq_stride = page_kv_cache.block_seq_stride
    transformer_block_count = config.transformer_block_count
    cache_count = 2

    return [
        block_seq_stride
        * attn_head_dim
        * attn_head_count
        * transformer_block_count
        * cache_count
    ]


class IreeInstance:
    def __init__(
        self,
        devices: list[str],
        vmfb: pathlib.Path | bytes,
        parameters: pathlib.Path | ParameterIndex,
        config: LlamaModelConfig | None = None,
    ):
        self._instance = iree.runtime.VmInstance()
        self._devices = [iree.runtime.get_device(d) for d in devices]
        self._iree_runtime_config = iree.runtime.Config(device=self._devices[0])

        if isinstance(vmfb, pathlib.Path | str):
            with open(vmfb, "rb") as f:
                vmfb = f.read()

        if config is None:
            assert not isinstance(
                parameters, ParameterIndex
            ), "Loading a LlamaModelConfig directly form iree.runtime.ParameterIndex is not supported yet. It needs to be provided separately"
            config = LlamaModelConfig.from_dataset(Dataset.load(parameters))
        self._config = config

        if not isinstance(parameters, ParameterIndex):
            paramIndex = iree.runtime.ParameterIndex()
            with open(str(parameters), "rb") as f:
                paramIndex.load_from_file_handle(
                    iree.runtime.FileHandle.wrap_fd(f.fileno()), "irpa"
                )
            parameters = paramIndex

        provider = parameters.create_provider("model")
        self._parameters = iree.runtime.create_io_parameters_module(
            self._instance, provider
        )

        self._hal = iree.runtime.create_hal_module(
            self._instance, devices=self._devices
        )
        self._binary = iree.runtime.VmModule.copy_buffer(self._instance, vmfb)
        self._modules = iree.runtime.load_vm_modules(
            self._parameters, self._hal, self._binary, config=self._iree_runtime_config
        )
        self._main_module = self._modules[-1]

        self._prefill = None
        self._decode = None

        # Grab the non-async functions:
        for funcname in self._main_module.vm_module.function_names:
            if "$async" not in funcname:
                func = self._main_module[funcname]
                setattr(self, funcname, func)
                if "prefill_bs" in funcname:
                    self._prefill = func
                    self.prefill_bs = int(funcname[10:])
                if "decode_bs" in funcname:
                    self._decode = func
                    self.decode_bs = int(funcname[9:])

        assert self._prefill is not None
        assert self._decode is not None

    def allocate(self, *shape, dtype, device_index: int) -> iree.runtime.DeviceArray:
        dtype = np_dtype_to_hal_dtype[dtype]

        device = self._devices[device_index]
        buffer = device.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=(iree.runtime.BufferUsage.DEFAULT),
            allocation_size=math.prod(shape) * 2,
        )

        buffer_view = iree.runtime.HalBufferView(
            buffer, shape=shape, element_type=dtype
        )
        return iree.runtime.DeviceArray(device=device, buffer_view=buffer_view)

    @property
    def config(self):
        return self._config

    def prefill(self, *args):
        results = self._prefill(*args)
        results = [numpy.asarray(r) for r in results]
        return results

    def decode(self, *args):
        results = self._decode(*args)
        results = [numpy.asarray(r) for r in results]
        return results


class TorchInstance:
    def __init__(
        self,
        theta: Theta,
        config: LlamaModelConfig,
        device: torch.device = None,
        prefill_bs: int = 1,
        decode_bs: int = 1,
        has_prefill_position: bool = False,
    ):
        self._model = PagedLlmModelV1(theta=theta, config=config)
        self.prefill_bs = prefill_bs
        self.decode_bs = decode_bs
        self._has_prefill_position = has_prefill_position
        self._device = device
        self._config = config

    @property
    def config(self):
        return self._config

    @staticmethod
    def load(
        filepath: pathlib.Path,
        device: torch.device | str = None,
        has_prefill_position: bool = False,
    ):
        dataset = Dataset.load(path=filepath, device=device)
        config = LlamaModelConfig.from_properties(dataset.properties)
        return TorchInstance(
            theta=dataset.root_theta,
            config=config,
            has_prefill_position=has_prefill_position,
        )

    def prefill(self, *args):
        # In order for the signature of this method to look like the exported MLIR
        # we need to handle the 2 different cases.
        if self._has_prefill_position:
            tokens = args[0]
            start_positions = args[1]
            seq_lens = args[2]
            seq_block_ids = args[3]
            cache_state = args[4:]
        else:
            tokens = args[0]
            start_positions = None
            seq_lens = args[1]
            seq_block_ids = args[2]
            cache_state = args[3:]

        tokens = torch.asarray(tokens, device=self._device)
        seq_lens = torch.asarray(seq_lens, device=self._device)
        seq_block_ids = torch.asarray(seq_block_ids, device=self._device)
        cache_state = [torch.asarray(cs, device=self._device) for cs in cache_state]
        if start_positions is not None:
            start_positions = torch.asarray(start_positions, device=self._device)

        logits = self._model.prefill(
            tokens,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
            start_positions=start_positions,
        )

        logits = unbox_tensor(logits)

        # TODO: This should be handled by the model
        logits = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        logits = torch.log(logits)

        logits = logits.cpu()

        return logits

    def decode(self, tokens, seq_lens, start_positions, seq_block_ids, *cache_state):
        tokens = torch.asarray(tokens, device=self._device)
        seq_lens = torch.asarray(seq_lens, device=self._device)
        start_positions = torch.asarray(start_positions, device=self._device)
        seq_block_ids = torch.asarray(seq_block_ids, device=self._device)
        cache_state = [torch.asarray(cs, device=self._device) for cs in cache_state]

        logits = self._model.decode(
            tokens,
            seq_lens=seq_lens,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )

        logits = ops.unshard(logits)

        # TODO: This should be handled by the model
        logits = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        logits = torch.log(logits)

        logits = logits.cpu()
        return logits

    def allocate(self, *shape, dtype, device_index: int):
        dtype = np_dtype_to_torch_dtype[dtype]
        return torch.zeros(*shape, dtype=dtype, device=self._device)


class LlmAllocator:
    def __init__(self, page_count, block_stride):
        self._pages = list(range(1, page_count))
        self._block_stride = block_stride

    def allocate(
        self, *, token_count: int | None = None, page_count: int | None = None
    ) -> list[int]:
        if token_count is not None:
            page_count = int(numpy.ceil(token_count / self._block_stride))

        assert page_count is not None
        assert (
            len(self._pages) >= page_count
        ), "Not enough free pages. The allocator may have been constructed with fewer pages than required or pages were not freed."

        pages = self._pages[:page_count]
        self._pages = self._pages[page_count:]
        return pages

    def free(self, pages: list[int]):
        self._pages.extend(pages)


def make_chunks(
    request: LlmRequest,
    chunk_block_size: int,
    block_stride: int,
) -> List[LlmTaskInput]:
    chunk_token_size = chunk_block_size * block_stride
    block_count = int(numpy.ceil(len(request.tokens) / block_stride))

    task_inputs = []
    for i in range(0, block_count, chunk_block_size):
        start_position = i * block_stride

        chunk_page_ids = request.pages[: i + chunk_block_size]
        chunk_tokens = request.tokens[
            start_position : start_position + chunk_token_size
        ]
        seq_len = start_position + len(chunk_tokens)

        task_inputs.append(
            LlmTaskInput(
                request_id=request.request_id,
                chunk_id=i // chunk_block_size,
                tokens=chunk_tokens,
                seq_len=seq_len,
                pages=chunk_page_ids,
                start_position=start_position,
            )
        )

    return task_inputs


class LlmRunner:
    def __init__(
        self,
        instance: IreeInstance | TorchInstance,
        page_count: int,
        page_sizes: list[int],
        block_stride: int,
        kv_cache_dtype: str,
        decode_topk_logits: int | None = 8,
        chunk_block_size: int | None = None,
    ):
        self._instance = instance
        self._page_count = page_count
        self._page_sizes = page_sizes
        self._block_stride = block_stride
        self._prefill_bs = instance.prefill_bs
        self._decode_bs = instance.decode_bs
        self._decode_topk_logits = decode_topk_logits
        self._chunk_block_size = chunk_block_size

        self._allocator = LlmAllocator(page_count=page_count, block_stride=block_stride)

        if chunk_block_size is not None:
            self._prefill_scheduler: Scheduler = ChunkScheduler(
                batch_size=self._prefill_bs,
                block_seq_stride=self._block_stride,
                llm_task_class=PrefillTask,
                invocation_fn=self._instance.prefill,
                chunk_block_size=chunk_block_size,
                has_prefill_position=True,
            )
        else:
            self._prefill_scheduler: Scheduler = BasicScheduler(
                batch_size=self._prefill_bs,
                block_seq_stride=self._block_stride,
                llm_task_class=PrefillTask,
                invocation_fn=self._instance.prefill,
            )

        self._decode_scheduler: Scheduler = BasicScheduler(
            batch_size=self._decode_bs,
            block_seq_stride=self._block_stride,
            llm_task_class=DecodeTask,
            invocation_fn=self._instance.decode,
        )

        self._cache = [
            instance.allocate(
                page_count,
                page_size,
                dtype=dtype_string_to_type[kv_cache_dtype],
                device_index=i,
            )
            for i, page_size in enumerate(page_sizes)
        ]

    def allocate(
        self, *, page_count: int | None = None, token_count: int | None = None
    ):
        return self._allocator.allocate(token_count=token_count, page_count=page_count)

    def free(self, pages: list[int]):
        self._allocator.free(pages)

    def make_requests(
        self, requests: List[List[int]], page_ids: List[List[int]]
    ) -> List[LlmRequest]:
        assert len(requests) == len(page_ids)

        llm_requests = []
        for i, (req, pages) in enumerate(zip(requests, page_ids)):
            llm_requests.append(
                LlmRequest(request_id=f"req-{i}", tokens=req, pages=pages)
            )

        return llm_requests

    def _make_prefill_task_inputs(
        self,
        request: LlmRequest,
    ) -> List[LlmTaskInput]:
        if self._chunk_block_size is None:
            return [
                LlmTaskInput(
                    request_id=request.request_id,
                    chunk_id=0,
                    tokens=request.tokens,
                    seq_len=len(request.tokens),
                    pages=request.pages,
                )
            ]

        return make_chunks(
            request=request,
            chunk_block_size=self._chunk_block_size,
            block_stride=self._block_stride,
        )

    def submit_prefill(
        self,
        requests: List[LlmRequest],
    ) -> List[LlmTaskInput]:
        task_inputs = []
        for request in requests:
            task_inputs.extend(self._make_prefill_task_inputs(request))

        for task in task_inputs:
            self._prefill_scheduler.schedule_task(task)

    def submit_decode(self, requests: List[LlmRequest]):
        task_inputs = []
        for request in requests:
            task_inputs.append(
                LlmTaskInput(
                    request_id=request.request_id,
                    chunk_id=0,
                    tokens=[request.tokens[-1]],
                    seq_len=len(request.tokens),
                    start_position=len(request.tokens) - 1,
                    pages=request.pages,
                )
            )

        for task_input in task_inputs:
            self._decode_scheduler.schedule_task(task_input)

    def run_prefill(
        self,
        selection_fn: SelectionFn,
    ):
        return self._prefill_scheduler.run(
            selection_fn=selection_fn,
            cache=self._cache,
        )

    def run_decode(
        self,
        selection_fn: SelectionFn,
    ):
        return self._decode_scheduler.run(
            selection_fn=selection_fn,
            cache=self._cache,
        )

    def prefill(
        self, requests: list[list[int]], page_ids: list[list[int]]
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        assert len(requests) == len(page_ids)

        # We want to return the logits and indices without doing an actual selection
        # of some token.
        def selection_fn(
            logits: numpy.ndarray, indices: numpy.ndarray, positions
        ) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
            return [(l, i) for l, i in zip(logits, indices, strict=True)]

        llm_requests = self.make_requests(requests=requests, page_ids=page_ids)
        self.submit_prefill(llm_requests)
        request_results: dict[
            str, list[tuple[numpy.ndarray, numpy.ndarray]]
        ] = self.run_prefill(selection_fn=selection_fn)

        if self._chunk_block_size is not None:
            # Stitch together resulting chunk logits and their respective vocabulary
            # indices.
            chunked_request_results = request_results
            request_results: dict[str, list[tuple[numpy.ndarray, numpy.ndarray]]] = {}
            for request_id, task_results in chunked_request_results.items():
                logits = [task_result[0] for task_result in task_results]
                logits = numpy.concatenate(logits)
                indices = [task_result[1] for task_result in task_results]
                indices = numpy.concatenate(indices)
                request_results[request_id] = [(logits, indices)]

        assert all(
            llm_request.request_id == request_result_key
            for llm_request, request_result_key in zip(
                llm_requests, request_results.keys(), strict=True
            )
        ), "prefill must return request in the same order as given"

        logits, indices = zip(*[(r[0][0], r[0][1]) for r in request_results.values()])
        return logits, indices

    def decode(
        self, tokens: list[int], positions: list[int], page_ids: list[list[int]]
    ):
        assert len(tokens) == len(positions)
        assert len(tokens) == len(page_ids)

        task_inputs = []
        for i, token in enumerate(tokens):
            task_inputs.append(
                LlmTaskInput(
                    request_id=f"req-{i}",
                    chunk_id=0,
                    tokens=[token],
                    seq_len=positions[i] + 1,
                    start_position=positions[i],
                    pages=page_ids[i],
                )
            )

        decode_task = DecodeTask(
            invocation_fn=self._instance.decode,
            llm_task_inputs=task_inputs,
            batch_size=self._decode_bs,
            block_stride=self._block_stride,
            decode_topk_logits=self._decode_topk_logits,
        )
        logits, indices = decode_task.run(*self._cache)
        return logits, indices


class LlmDecoder:
    def __init__(self, runner: LlmRunner):
        self._runner = runner

    def _greedy_select(self, logits, indices, positions):
        selected = []
        argmax = numpy.argmax(logits, axis=-1)
        for i, pos in enumerate(positions):
            token = argmax[i][pos]
            if indices is not None:
                token = indices[i][pos][token]
            selected.append(token)

        return selected

    def greedy_decode(
        self, requests: list[list[int]], steps: int, eos: int | None = None
    ):
        done = {}
        requests_map = {}

        prompt_lengths = [len(req) for req in requests]
        page_ids = [
            self._runner.allocate(token_count=len(req) + steps) for req in requests
        ]

        llm_requests = self._runner.make_requests(requests, page_ids)
        # Make sure we don't modify the arguments of this function as we later add
        # decoded tokens to the lists.
        llm_requests = deepcopy(llm_requests)
        for req in llm_requests:
            requests_map[req.request_id] = req
            done[req.request_id] = False

        self._runner.submit_prefill(llm_requests)

        selections = self._runner.run_prefill(self._greedy_select)
        for rid, tokens in selections.items():
            token = tokens[0]
            request = requests_map[rid]
            if token == eos:
                done[rid] = True
                continue

            request.tokens.append(token)

        for _ in range(steps - 1):
            if all(list(done.values())):
                break
            to_submit = [
                request for request in llm_requests if not done[request.request_id]
            ]
            self._runner.submit_decode(to_submit)
            selections = self._runner.run_decode(self._greedy_select)
            for rid, tokens in selections.items():
                token = tokens[0]
                request = requests_map[rid]
                if token == eos:
                    done[rid] = True
                    continue

                request.tokens.append(token)

        return [
            request.tokens[prompt_lengths[index] :]
            for index, request in enumerate(llm_requests)
        ]


class LlmBencher:
    @dataclasses.dataclass
    class BenchResults:
        samples_per_sec: float
        bs: int
        total_ms: float
        prefill_ms: float
        decode_ms: float
        decode_step_ms: float

    def __init__(self, batch: LlmRunner):
        self._batch = batch

    def greedy_bench(self, length: int, steps: int):
        prefill_bs = self._batch._prefill_bs
        decode_bs = self._batch._decode_bs

        prefill_requests = [[0] * length] * prefill_bs
        decode_request = [0] * decode_bs
        positions = [length] * decode_bs

        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        for _ in range(int(numpy.ceil(decode_bs / prefill_bs))):
            _, _ = self._batch.prefill(prefill_requests)

        prefill = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        for _ in range(steps - 1):
            positions = [p + 1 for p in positions]
            self._batch.decode(tokens=decode_request, positions=positions)

        decode = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        # Compute the total runtime
        total = decode - start
        prefill = prefill - start
        decode = total - prefill

        # Convert to ms
        total = total * 1e-6
        prefill = prefill * 1e-6
        decode = decode * 1e-6
        decode_step = decode / steps

        results = self.BenchResults(
            samples_per_sec=decode_bs / total * 1e3,
            bs=decode_bs,
            total_ms=total,
            prefill_ms=prefill,
            decode_ms=decode,
            decode_step_ms=decode_step,
        )

        return results


class LlmPerplexityEval:
    @dataclasses.dataclass
    class Dataset:
        dataset: str
        revision: str
        split: str
        ids: list[int]
        scores: dict[int, float] | None = None

        def as_dict(self):
            return dataclasses.asdict(self)

        def compare(self, other):
            assert self.dataset == other.dataset
            assert self.revision == other.revision
            assert self.split == other.split
            diff = [self.scores[str(id)] - other.scores[str(id)] for id in self.ids]
            has_nan = any([math.isnan(d) for d in diff])
            if has_nan:
                return math.inf

            return max([math.fabs(d) for d in diff])

    @dataclasses.dataclass
    class Result:
        valid: bool
        score: float

    def __init__(self, batch: LlmRunner, logits_normalization: str):
        self._batch = batch
        self._logits_normalization = logits_normalization

    def compute_cross_entropy(
        self,
        logits: Sequence[numpy.ndarray],
        indices: Sequence[numpy.ndarray],
        requests: list[list[int]],
        min_context: int = 0,
    ) -> list["LlmPerplexityEval.Result"]:
        results = []
        for i, req in enumerate(requests):
            req_len = len(req)
            ctx_len = req_len - 1
            in_indices = numpy.asarray(req[1:])
            req_logits = logits[i][:ctx_len]
            req_indices = indices[i][: req_len - 1]

            if self._logits_normalization == "none":
                req_logits = numpy.asarray(req_logits, dtype=numpy.float32)
                req_logits = numpy.exp(req_logits)
                req_logits = req_logits / numpy.sum(req_logits, axis=-1, keepdims=True)
                req_logits = numpy.log(req_logits)
            elif self._logits_normalization == "softmax":
                req_logits = numpy.asarray(req_logits, dtype=numpy.float32)
                req_logits = numpy.log(req_logits)
            elif self._logits_normalization != "log_softmax":
                raise ValueError(
                    f"Unknown logits normalization: {self._logits_normalization}"
                )

            matches = in_indices[:, None] == req_indices
            all_available = (numpy.sum(matches) == ctx_len).item()
            scores = numpy.sum(numpy.where(matches, req_logits, 0.0), axis=-1)

            scores = scores[min_context:]
            ctx_len = ctx_len - min_context

            err = (-numpy.sum(scores) / ctx_len).item()

            results.append(LlmPerplexityEval.Result(all_available, err))

        return results

    @property
    def prefill_bs(self):
        return self._batch._prefill_bs

    @property
    def decode_bs(self):
        return self._batch._decode_bs

    def prefill_cross_entropy(
        self, requests: list[list[int]], **kwargs
    ) -> list["LlmPerplexityEval.Result"]:
        page_ids = [self._batch.allocate(token_count=len(req)) for req in requests]
        logits, indices = self._batch.prefill(requests, page_ids=page_ids)
        flat_page_ids = list(itertools.chain.from_iterable(page_ids))
        self._batch.free(flat_page_ids)
        return self.compute_cross_entropy(logits, indices, requests, **kwargs)

    def decode_cross_entropy(self, requests: list[list[int]]):
        page_ids = [self._batch.allocate(token_count=len(req)) for req in requests]

        sl = max(len(req) for req in requests)

        steps = [[0 for _ in range(len(requests))] for _ in range(sl)]
        for i, req in enumerate(requests):
            for j, t in enumerate(req):
                steps[j][i] = t

        logits = []
        indices = []
        for i, step in enumerate(steps):
            pos = [i] * len(requests)
            logit, ind = self._batch.decode(step, pos, page_ids=page_ids)
            logits.append(logit)
            indices.append(ind)

        flat_page_ids = list(itertools.chain.from_iterable(page_ids))
        self._batch.free(flat_page_ids)

        logits = numpy.concatenate(logits, axis=1)
        indices = numpy.concatenate(indices, axis=1)
        return self.compute_cross_entropy(logits, indices, requests)

    def batch_prefill_perplexity(self, requests: list[list[int]], **kwargs):
        bs = self.prefill_bs
        results = []
        while len(requests) > 0:
            batch = requests[:bs]
            requests = requests[bs:]
            cross_entropy = self.prefill_cross_entropy(requests=batch, **kwargs)
            results.extend(cross_entropy)
        return results

    def run_dataset(self, dataset: Dataset, tokenizer, **kwargs):
        name = dataset.dataset
        revision = dataset.revision
        split = dataset.split
        ids = dataset.ids

        test_prompts = load_dataset(name, revision, split=split)["text"]
        test_prompts = [test_prompts[id] for id in ids]
        encoded, lens = tokenizer.encode(test_prompts)
        encoded = [ids[:len] for ids, len in zip(encoded, lens)]

        results = self.batch_prefill_perplexity(requests=encoded, **kwargs)

        scores = {str(id): result.score for id, result in zip(ids, results)}
        return self.Dataset(
            dataset=name, revision=revision, split=split, ids=ids, scores=scores
        )


class LlmInstance:
    def __init__(
        self,
        model_instance,
        block_seq_stride,
        page_sizes: list[int],
        block_count,
        logits_normalization="log_softmax",
        kv_cache_dtype="float16",
        decode_topk_logits: int | None = 8,
        chunk_block_size: int | None = None,
    ):
        self._instance = model_instance
        self._block_seq_stride = block_seq_stride
        self._page_sizes = page_sizes
        self._block_count = block_count
        self.kv_cache_dtype = kv_cache_dtype
        self._logits_normalization = logits_normalization
        self._decode_topk_logits = decode_topk_logits
        self._chunk_block_size = chunk_block_size

    @staticmethod
    def load(instance, config: ServiceConfig, chunk_block_size: Optional[int] = None):
        page_kv_cache = config.paged_kv_cache
        _block_seq_stride = page_kv_cache.block_seq_stride
        _block_count = page_kv_cache.device_block_count
        _logits_normalization = config.logits_normalization
        _page_sizes = server_config_page_size(config)

        return LlmInstance(
            model_instance=instance,
            block_count=_block_count,
            block_seq_stride=_block_seq_stride,
            page_sizes=_page_sizes,
            logits_normalization=_logits_normalization,
            kv_cache_dtype=page_kv_cache.kv_cache_dtype,
            chunk_block_size=chunk_block_size,
        )

    def make_runner(self):
        return LlmRunner(
            instance=self._instance,
            page_count=self._block_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_seq_stride,
            kv_cache_dtype=self.kv_cache_dtype,
            decode_topk_logits=self._decode_topk_logits,
            chunk_block_size=self._chunk_block_size,
        )

    def make_bencher(self):
        return LlmBencher(self.make_runner())

    def make_decoder(self):
        return LlmDecoder(self.make_runner())

    def make_perplexity_eval(self):
        return LlmPerplexityEval(
            self.make_runner(), logits_normalization=self._logits_normalization
        )
