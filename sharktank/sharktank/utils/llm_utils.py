import iree.runtime
import math
import numpy
import torch

from iree.runtime import ParameterIndex
from sharktank.layers.configs.llm_configs import LlamaModelConfig
from sharktank.models.llm import PagedLlmModelV1
from sharktank.types import Theta

np_dtype_to_torch_dtype = {
    numpy.float16: torch.float16,
    numpy.float32: torch.float32,
}

np_dtype_to_hal_dtype = {
    numpy.float16: iree.runtime.HalElementType.FLOAT_16,
    numpy.float32: iree.runtime.HalElementType.FLOAT_32,
}


def llama_config_page_size(config: LlamaModelConfig):
    return (
        config.hp.attention_head_count_kv
        * config.hp.attn_head_dim
        * config.hp.block_count
        * config.block_seq_stride
        * 2
    )


def server_config_page_size(config: dict):
    page_kv_cache = config["paged_kv_cache"]
    attn_head_dim = config["attn_head_dim"]
    attn_head_count = page_kv_cache["attention_head_count_kv"]
    block_seq_stride = page_kv_cache["block_seq_stride"]
    transformer_block_count = config["transformer_block_count"]
    cache_count = 2

    return (
        block_seq_stride
        * attn_head_dim
        * attn_head_count
        * transformer_block_count
        * cache_count
    )


class IreeInstance:
    def __init__(self, devices: list[str], vmfb: bytes, parameters: ParameterIndex):

        self._instance = iree.runtime.VmInstance()
        self._devices = [iree.runtime.get_device(d) for d in devices]
        self._config = iree.runtime.Config(device=self._devices[0])

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
    ):
        self._instance = instance
        self._page_count = page_count
        self._page_size = page_size
        self._block_stride = block_stride
        self._prefill_bs = instance._prefill_bs
        self._decode_bs = instance._decode_bs

        self._cache = instance.allocate(page_count, page_size, dtype=numpy.float16)
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
        else:
            k = 8
            logits = torch.asarray(numpy.asarray(results))
            logits, indices = torch.topk(logits, k)

        logits = numpy.asarray(logits)
        indices = numpy.asarray(indices)

        return logits, indices

    def decode(self, tokens: list[int], positions: list[int]):
        assert self._bs == len(tokens)
        assert self._bs == len(positions)

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
            ind = argmax[i][pos]
            token = indices[i][pos][ind]
            selected.append(token)

        return selected

    def greedy_decode(self, requests: list[list[int]], steps: int):
        selections = []
        positions = [len(request) - 1 for request in requests]

        logits, indices = self._batch.prefill(requests)
        last = self._greedy_select(logits, indices, positions)
        selections.append(last)

        for _ in range(steps - 1):
            positions = [p + 1 for p in positions]
            logits, indices = self._batch.decode(tokens=last, positions=positions)
            last = self._greedy_select(logits, indices, [0] * len(requests))
            selections.append(last)

        results = [[] for i in range(len(selections[0]))]
        for select in selections:
            for j, token in enumerate(select):
                results[j].append(token.item())

        return results


class LlmPerplexityEval:
    def __init__(self, batch):
        self._batch = batch

    @staticmethod
    def compute_cross_entropy(logits, indices, requests):
        results = []
        requests = torch.asarray(requests)
        for i, req in enumerate(requests):
            req_len = len(req)
            in_indices = req[1:]
            req_logits = logits[i, : req_len - 1]
            req_indices = indices[i, : req_len - 1]

            matches = in_indices[:, None] == req_indices

            all_available = torch.sum(matches) == req_len - 1
            scores = numpy.sum(numpy.where(matches, req_logits, 0.0), axis=-1)
            cross_entropy = -numpy.sum(scores) / (req_len - 1)
            results.append((all_available, cross_entropy))

        return results

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


class LlmInstance:
    def __init__(self, model_instance, block_seq_stride, page_size, block_count):
        self._instance = model_instance
        self._block_seq_stride = block_seq_stride
        self._page_size = page_size
        self._block_count = block_count

    def make_batch(self):
        return LlmBatch(
            instance=self._instance,
            page_count=self._block_count,
            page_size=self._page_size,
            block_stride=self._block_seq_stride,
        )

    def make_decoder(self):
        return LlmDecoder(self.make_batch())

    def make_perplexity_eval(self):
        return LlmPerplexityEval(self.make_batch())
