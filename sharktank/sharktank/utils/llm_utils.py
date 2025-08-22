import dataclasses
import iree.runtime
import math
import numpy
import pathlib
import time
import torch

from iree.runtime import ParameterIndex
from sharktank.layers.configs.llm_configs import LlamaModelConfig
from sharktank.models.llm.config import ServiceConfig
from sharktank.models.llm import PagedLlmModelV1
from sharktank.types import Theta

np_dtype_to_torch_dtype = {
    numpy.float16: torch.float16,
    numpy.float32: torch.float32,
}

np_dtype_to_hal_dtype = {
    numpy.float16: iree.runtime.HalElementType.FLOAT_16,
    numpy.float32: iree.runtime.HalElementType.FLOAT_32,
    torch.float8_e4m3fn: iree.runtime.HalElementType.FLOAT_8_E4M3_FN,
    torch.float8_e4m3fnuz: iree.runtime.HalElementType.FLOAT_8_E4M3_FNUZ,
}

dtype_string_to_type = {
    "float16": numpy.float16,
    "float32": numpy.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
}


def llama_config_page_size(config: LlamaModelConfig):
    return (
        config.hp.attention_head_count_kv
        * config.hp.attn_head_dim
        * config.hp.block_count
        * config.block_seq_stride
        * 2
    )


def server_config_page_size(config: ServiceConfig):
    page_kv_cache = config.paged_kv_cache
    attn_head_dim = config.attn_head_dim
    attn_head_count = page_kv_cache.attention_head_count_kv
    block_seq_stride = page_kv_cache.block_seq_stride
    transformer_block_count = config.transformer_block_count
    cache_count = 2

    return (
        block_seq_stride
        * attn_head_dim
        * attn_head_count
        * transformer_block_count
        * cache_count
    )


class IreeInstance:
    def __init__(
        self,
        devices: list[str],
        vmfb: pathlib.Path | bytes,
        parameters: pathlib.Path | ParameterIndex,
    ):

        self._instance = iree.runtime.VmInstance()
        self._devices = [iree.runtime.get_device(d) for d in devices]
        self._config = iree.runtime.Config(device=self._devices[0])

        if isinstance(vmfb, pathlib.Path | str):
            with open(vmfb, "rb") as f:
                vmfb = f.read()

        if not isinstance(parameters, ParameterIndex):
            paramIndex = iree.runtime.ParameterIndex()
            paramIndex.load(parameters)
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
            self._parameters, self._hal, self._binary, config=self._config
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
                    self._prefill_bs = int(funcname[10:])
                if "decode_bs" in funcname:
                    self._decode = func
                    self._decode_bs = int(funcname[9:])

        assert self._prefill is not None
        assert self._decode is not None

    def allocate(self, *shape, dtype):
        dtype = np_dtype_to_hal_dtype[dtype]

        device = self._devices[0]
        buffer = device.allocator.allocate_buffer(
            memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
            allowed_usage=(iree.runtime.BufferUsage.DEFAULT),
            allocation_size=math.prod(shape) * 2,
        )

        buffer_view = iree.runtime.HalBufferView(
            buffer, shape=shape, element_type=dtype
        )
        return iree.runtime.DeviceArray(device=device, buffer_view=buffer_view)

    def prefill(self, *args):
        return self._prefill(*args)

    def decode(self, *args):
        return self._decode(*args)


class TorchInstance:
    def __init__(self, theta: Theta, config: LlamaModelConfig):
        self._model = PagedLlmModelV1(theta=theta, config=config)
        self._prefill_bs = 1
        self._decode_bs = 1

    def prefill(self, tokens, seq_lens, seq_block_ids, cache_state):
        tokens = torch.asarray(tokens)
        seq_lens = torch.asarray(seq_lens)
        seq_block_ids = torch.asarray(seq_block_ids)
        cache_state = [torch.asarray(cache_state)]

        sl = tokens.shape[1]
        input_mask = self._model.input_mask(seq_lens, sl)
        attention_mask = self._model.attention_mask(input_mask)

        logits = self._model.prefill(
            tokens,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )

        # TODO: This should be handled by the model
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = torch.log(logits)
        k = 8
        logits, indices = torch.topk(logits, k)

        return logits, indices

    def decode(self, tokens, seq_lens, start_positions, seq_block_ids, cache_state):
        tokens = torch.asarray(tokens)
        seq_lens = torch.asarray(seq_lens)
        start_positions = torch.asarray(start_positions)
        seq_block_ids = torch.asarray(seq_block_ids)
        cache_state = [torch.asarray(cache_state)]

        input_mask = self._model.input_mask(
            seq_lens, seq_block_ids.shape[1] * self._model.cache.block_seq_stride
        )
        attention_mask = self._model.decode_attention_mask(input_mask)

        logits = self._model.decode(
            tokens,
            attention_mask=attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )

        # TODO: This should be handled by the model
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = torch.log(logits)
        k = 8
        logits, indices = torch.topk(logits, k)

        return logits, indices

    def allocate(self, *shape, dtype):
        dtype = np_dtype_to_torch_dtype[dtype]
        return torch.zeros(*shape, dtype=dtype)


class LlmBatch:
    def __init__(
        self,
        instance: IreeInstance,
        page_count: int,
        page_size: int,
        block_stride: int,
        kv_cache_dtype: str,
    ):
        self._instance = instance
        self._page_count = page_count
        self._page_size = page_size
        self._block_stride = block_stride
        self._prefill_bs = instance._prefill_bs
        self._decode_bs = instance._decode_bs

        self._cache = instance.allocate(
            page_count, page_size, dtype=dtype_string_to_type[kv_cache_dtype]
        )
        self._page_id = 1

    def reset(self, bs):
        self._bs = bs
        assert self._bs <= self._prefill_bs
        assert self._bs <= self._decode_bs

    def get_pages(self, bs: int, count: int):
        pages = numpy.arange(start=1, stop=bs * count + 1, dtype=numpy.int64)
        pages = pages.reshape(count, bs).T
        return pages

    def prefill(self, requests: list[list[int]]):
        self.reset(len(requests))

        max_len = max(len(request) for request in requests)
        blocks = math.ceil(max_len / self._block_stride)
        blocked_len = blocks * self._block_stride

        tokens = numpy.zeros((self._prefill_bs, blocked_len), dtype=numpy.int64)
        lens = numpy.ones((self._prefill_bs,), dtype=numpy.int64)
        pages = numpy.zeros((self._prefill_bs, blocks), dtype=numpy.int64)

        for i, request in enumerate(requests):
            tokens[i, : len(request)] = request
            lens[i] = len(request)

        pages[: self._bs, :] = self.get_pages(self._bs, blocks)

        results = self._instance.prefill(tokens, lens, pages, self._cache)

        if isinstance(results, tuple):
            logits, indices = results
            logits = numpy.asarray(logits)
            indices = numpy.asarray(indices)
        else:
            logits = numpy.asarray(results)
            indices = None

        return logits, indices

    def decode(self, tokens: list[int], positions: list[int]):
        assert len(tokens) == len(positions)

        max_len = max(positions) + 1
        blocks = math.ceil(max_len / self._block_stride)

        tokens_ = numpy.zeros((self._decode_bs, 1), dtype=numpy.int64)
        lens_ = numpy.ones((self._decode_bs,), dtype=numpy.int64)
        pos_ = numpy.ones((self._decode_bs,), dtype=numpy.int64)
        pages_ = numpy.zeros((self._decode_bs, blocks), dtype=numpy.int64)

        for i in range(self._bs):
            tokens_[i, 0] = tokens[i]
            lens_[i] = positions[i] + 1
            pos_[i] = positions[i]

        pages_[: self._bs, :] = self.get_pages(self._bs, blocks)

        results = self._instance.decode(tokens_, lens_, pos_, pages_, self._cache)

        if isinstance(results, tuple):
            logits, indices = results
        else:
            k = 8
            logits = torch.asarray(numpy.asarray(results))
            logits, indices = torch.topk(logits, k)

        logits = numpy.asarray(logits)
        indices = numpy.asarray(indices)

        return logits, indices


class LlmDecoder:
    def __init__(self, batch):
        self._batch = batch

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
        selections = []
        positions = [len(request) - 1 for request in requests]

        logits, indices = self._batch.prefill(requests)
        last = self._greedy_select(logits, indices, positions)
        done = [False for _ in range(len(requests))]
        done = [d or t == eos for d, t in zip(done, last)]

        selections.append(last)

        for _ in range(steps - 1):
            if all(done):
                break
            positions = [p + 1 for p in positions]
            logits, indices = self._batch.decode(tokens=last, positions=positions)
            last = self._greedy_select(logits, indices, [0] * len(requests))
            done = [d or t == eos for d, t in zip(done, last)]
            selections.append(last)

        results = [[] for i in range(len(selections[0]))]
        for select in selections:
            for j, token in enumerate(select):
                results[j].append(token.item())

        eos_pos = [[i for i, t in enumerate(result) if t == eos] for result in results]
        results = [
            result[: pos[0] + 1] if len(pos) > 0 else result
            for result, pos in zip(results, eos_pos)
        ]
        return results


class LlmBencher:
    @dataclasses.dataclass
    class BenchResults:
        samples_per_sec: float
        bs: int
        total_ms: float
        prefill_ms: float
        decode_ms: float
        decode_step_ms: float

    def __init__(self, batch: LlmBatch):
        self._batch = batch

    def greedy_bench(self, length: int, steps: int):
        prefill_bs = self._batch._prefill_bs
        decode_bs = self._batch._decode_bs

        prefill_requests = [[0] * length] * prefill_bs
        decode_request = [0] * decode_bs
        positions = [length] * decode_bs

        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        for _ in range(math.ceil(decode_bs / prefill_bs)):
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
    class Result:
        valid: bool
        score: float

    def __init__(self, batch, logits_normalization):
        self._batch = batch
        self._logits_normalization = logits_normalization

    def compute_cross_entropy(self, logits, indices, requests):
        results = []
        for i, req in enumerate(requests):
            req_len = len(req)
            in_indices = torch.asarray(req[1:])
            req_logits = logits[i, : req_len - 1]

            if indices is None:
                req_indices = torch.arange(req_logits.shape[-1])[None, None, :]
            else:
                req_indices = indices[i, : req_len - 1]

            matches = in_indices[:, None] == req_indices

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

            all_available = (torch.sum(matches) == req_len - 1).item()
            scores = numpy.sum(numpy.where(matches, req_logits, 0.0), axis=-1)
            cross_entropy = (-numpy.sum(scores) / (req_len - 1)).item()
            results.append(LlmPerplexityEval.Result(all_available, cross_entropy))

        return results

    @property
    def prefill_bs(self):
        return self._batch._prefill_bs

    @property
    def decode_bs(self):
        return self._batch._decode_bs

    def prefill_cross_entropy(self, requests: list[list[int]]):
        logits, indices = self._batch.prefill(requests)
        return self.compute_cross_entropy(logits, indices, requests)

    def decode_cross_entropy(self, requests: list[list[int]]):
        self._batch.reset(len(requests))

        sl = max(len(req) for req in requests)

        steps = [[0 for _ in range(len(requests))] for _ in range(sl)]
        for i, req in enumerate(requests):
            for j, t in enumerate(req):
                steps[j][i] = t

        logits = []
        indices = []
        for i, step in enumerate(steps):
            pos = [i] * len(requests)
            logit, ind = self._batch.decode(step, pos)
            logits.append(logit)
            indices.append(ind)

        logits = numpy.concatenate(logits, axis=1)
        indices = numpy.concatenate(indices, axis=1)
        return self.compute_cross_entropy(logits, indices, requests)

    def batch_prefill_perplexity(self, requests: list[list[int]]):
        bs = self.prefill_bs
        results = []
        while len(requests) > 0:
            batch = requests[:bs]
            requests = requests[bs:]
            cross_entropy = self.prefill_cross_entropy(requests=batch)
            results.extend(cross_entropy)
        return results


class LlmInstance:
    def __init__(
        self,
        model_instance,
        block_seq_stride,
        page_size,
        block_count,
        logits_normalization="log_softmax",
        kv_cache_dtype="float16",
    ):
        self._instance = model_instance
        self._block_seq_stride = block_seq_stride
        self._page_size = page_size
        self._block_count = block_count
        self.kv_cache_dtype = kv_cache_dtype
        self._logits_normalization = logits_normalization

    @staticmethod
    def load(instance, config: ServiceConfig):
        page_kv_cache = config.paged_kv_cache
        _block_seq_stride = page_kv_cache.block_seq_stride
        _block_count = page_kv_cache.device_block_count
        _logits_normalization = config.logits_normalization
        _page_size = server_config_page_size(config)

        return LlmInstance(
            model_instance=instance,
            block_count=_block_count,
            block_seq_stride=_block_seq_stride,
            page_size=_page_size,
            logits_normalization=_logits_normalization,
        )

    def make_batch(self):
        return LlmBatch(
            instance=self._instance,
            page_count=self._block_count,
            page_size=self._page_size,
            block_stride=self._block_seq_stride,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    def make_bencher(self):
        return LlmBencher(self.make_batch())

    def make_decoder(self):
        return LlmDecoder(self.make_batch())

    def make_perplexity_eval(self):
        return LlmPerplexityEval(
            self.make_batch(), logits_normalization=self._logits_normalization
        )
