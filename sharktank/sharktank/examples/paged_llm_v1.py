# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""


import math
from ..models.llama.tools.data_utils import write_ndarray_to_bin
import torch
import numpy as np
from ..layers import *
from ..types import *

from ..ops import replicate, unshard

# TODO: Should be using a base class with the protocol supported.
from ..models.mixtral.mixtral import *
from ..models.grok.grok import *
from ..models.llama.llama import *
from ..models.llama.sharding import shard_theta
from ..utils.debugging import trace_tensor
from ..utils.tokenizer import InferenceTokenizer


class TorchGenerator:
    """Generator that runs directly on the Torch model."""

    def __init__(
        self,
        model: PagedLlamaModelV1,
        tokenizer: InferenceTokenizer,
        page_cache_size: int = 128,
        # Need to look at the model more for this.
        end_token: int = 2,
        dump_bins: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shared_cache_state = model.cache.allocate(page_cache_size)
        self.free_pages = list(range(1, page_cache_size))
        self.end_token = end_token
        self.dump_bins = dump_bins

    @property
    def block_seq_stride(self) -> int:
        return self.model.cache.block_seq_stride

    def begin_batch(self, prompts: list[str]):
        token_ids, seq_lens = self.tokenizer.encode(
            prompts, pad_to_multiple_of=self.model.cache.pad_sequence_stride
        )

        token_ids = torch.tensor(token_ids, device=self.model.device)
        seq_lens = torch.tensor(seq_lens, device=self.model.device)
        if self.shared_cache_state is not None:
            cache_state = self.shared_cache_state
        else:
            cache_state = self.model.cache.allocate(bs=len(prompts))
        return Batch(self, token_ids, seq_lens, cache_state, dump_bins=self.dump_bins)

    def alloc_page(self) -> int:
        return self.free_pages.pop()

    def release_page(self, index: int):
        if self.model.config.kv_cache_type == "direct":
            return
        self.free_pages.append(index)


class Batch:
    def __init__(
        self,
        parent: TorchGenerator,
        token_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        cache_state: list[torch.Tensor],
        dump_bins: bool = False,
    ):
        self.bs = token_ids.shape[0]
        assert seq_lens.shape[0] == self.bs
        self.parent = parent
        self.token_ids = token_ids
        self.seq_lens = seq_lens
        self.cache_state = cache_state
        self.results: list[list[int]] = [[] for _ in range(self.bs)]
        self.done_result_indices: set[int] = set()
        self.dump_bins = dump_bins

        # Assemble the batch.
        seq_stride = self.parent.block_seq_stride
        self.seq_block_ids: list[list[int]] = []
        for seq_len in self.seq_lens:
            blocks_needed = (
                int(math.ceil(seq_len / seq_stride)) if seq_stride > 0 else 0
            )
            row = []
            for _ in range(blocks_needed):
                row.append(self.parent.alloc_page())
            self.seq_block_ids.append(row)

    @property
    def done(self) -> bool:
        return len(self.done_result_indices) == self.bs

    def detokenize(self) -> list[str]:
        return self.parent.tokenizer.decode(self.results)

    def print_current_results(self):
        results = self.detokenize()
        for i, s in enumerate(results):
            seq_len = int(self.seq_lens[i])
            print(f"  {i}({len(self.results[i])}, {seq_len}): {s}")

    def add_result_token(self, tokens: torch.Tensor):
        for i in range(self.bs):
            token = tokens[i][0]
            if token == self.parent.end_token:
                self.done_result_indices.add(i)
            if i in self.done_result_indices:
                continue
            token = int(tokens[i, 0])
            self.results[i].append(token)

    def allocate_seq_block_ids(self):
        for i in range(self.bs):
            sl = int(self.seq_lens[i])
            if (sl % self.parent.block_seq_stride) == 0:
                needed_blocks = sl // self.parent.block_seq_stride + 1
            else:
                needed_blocks = math.ceil(sl / self.parent.block_seq_stride)
            block_ids_row = self.seq_block_ids[i]
            while len(block_ids_row) < needed_blocks:
                block_ids_row.append(self.parent.alloc_page())

    def prefill(self):
        model = self.parent.model
        attention_mask = model.attention_mask(
            model.input_mask(self.seq_lens, self.token_ids.shape[1])
        )
        seq_block_ids_tensor = self.pad_block_ids()
        print(f":: Invoke prefill:")
        trace_tensor("prefill.token_ids", self.token_ids)
        trace_tensor("prefill.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("prefill.attention_mask", attention_mask)

        token_ids = self.token_ids
        if model.config.tensor_parallelism_size != 1:
            tp = model.config.tensor_parallelism_size
            token_ids = replicate(token_ids, tp)
            attention_mask = replicate(attention_mask, tp)
            seq_block_ids_tensor = replicate(seq_block_ids_tensor, tp)

        if self.dump_bins:
            write_ndarray_to_bin(
                token_ids.numpy(),
                f"prefill_token_ids_{'x'.join([str(x) for x in token_ids.shape])}xi64.bin",
            )
            write_ndarray_to_bin(
                np.array(token_ids.shape[0], dtype=np.int64),
                f"prefill_seq_lens_1xi64.bin",
            )
            write_ndarray_to_bin(
                seq_block_ids_tensor.numpy(),
                f"prefill_seq_block_ids_{'x'.join([str(x) for x in seq_block_ids_tensor.shape])}xi64.bin",
            )
            write_ndarray_to_bin(
                self.cache_state[0].to(torch.float8_e4m3fnuz).to(torch.uint8).numpy(),
                f"prefill_cache_state_{'x'.join([str(x) for x in self.cache_state[0].shape])}xf8E4M3FNUZ.bin",
            )
        logits = model.prefill(
            token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        logits = unshard(logits)

        # TODO: Generalize the sampling and don't make it swap on/off cpu.
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(logits, self.seq_lens)
        ).unsqueeze(1)
        print(f":: Prefill results:\n{tokens.tolist()}")
        self.add_result_token(tokens)
        self.next_tokens = tokens.to(device=model.device)

    def decode(self):
        model = self.parent.model
        start_positions = self.seq_lens.clone()
        self.seq_lens.add_(1)
        self.allocate_seq_block_ids()
        # TODO: Allocate more blocks on overflow.
        seq_block_ids_tensor = self.pad_block_ids()
        decode_attention_mask = model.decode_attention_mask(
            model.input_mask(
                self.seq_lens,
                seq_block_ids_tensor.shape[1] * self.parent.block_seq_stride,
            )
        )
        trace_tensor("decode.token_ids", self.next_tokens)
        trace_tensor("decode.start_positions", start_positions)
        trace_tensor("decode.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("decode.attention_mask", decode_attention_mask)

        if model.config.tensor_parallelism_size != 1:
            tp = model.config.tensor_parallelism_size
            self.next_tokens = replicate(self.next_tokens, tp)
            start_positions = replicate(start_positions, tp)
            seq_block_ids_tensor = replicate(seq_block_ids_tensor, tp)
            decode_attention_mask = replicate(decode_attention_mask, tp)

        if self.dump_bins:
            write_ndarray_to_bin(
                self.next_tokens.numpy(),
                f"decode_next_tokens_{'x'.join([str(x)for x in self.next_tokens.shape])}xi64.bin",
            )
            write_ndarray_to_bin(
                start_positions.numpy(),
                f"decode_start_positions_{'x'.join([str(x)for x in start_positions.shape])}xi64.bin",
            )
            write_ndarray_to_bin(
                seq_block_ids_tensor.numpy(),
                f"decode_seq_block_ids_tensor_{'x'.join([str(x)for x in seq_block_ids_tensor.shape])}xi64.bin",
            )
            write_ndarray_to_bin(
                torch.tensor(self.next_tokens.shape[0]).to(torch.int64).numpy(),
                f"decode_seq_lens_1xi64.bin",
            )
            write_ndarray_to_bin(
                self.cache_state[0].to(torch.float8_e4m3fnuz).to(torch.uint8).numpy(),
                f"decode_cache_state_{'x'.join([str(x) for x in self.cache_state[0].shape])}xf8E4M3FNUZ.bin",
            )
        logits = model.decode(
            self.next_tokens,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        logits = unshard(logits)
        trace_tensor("decode.logits", logits)
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(logits, [1] * self.bs),
            device=self.parent.model.device,
        ).unsqueeze(1)
        self.add_result_token(tokens)
        self.next_tokens = tokens

    def pad_block_ids(self) -> torch.Tensor:
        max_length = max(len(r) for r in self.seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in self.seq_block_ids]
        return torch.tensor(rows, device=self.parent.model.device)


def main():
    from ..utils import cli

    parser = cli.create_parser()
    parser.add_argument("prompt", nargs="+", help="Prompt strings")
    parser.add_argument(
        "--save_intermediates_path",
        help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors",
    )
    parser.add_argument(
        "--dump-bins",
        help="dump input tensors to bin files",
        action="store_true",
    )
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_quantization_options(parser)
    cli.add_model_options(parser)
    args = cli.parse(parser)
    device = torch.device(args.device) if args.device else None
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)
    prompts = args.prompt
    config = LlamaModelConfig(
        hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
        block_seq_stride=args.block_seq_stride,
        device=device,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        attention_kernel=args.attention_kernel,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        tensor_parallelism_size=args.tensor_parallelism_size,
        fake_quant=args.fake_quant,
    )
    if config.tensor_parallelism_size > 1:
        dataset.root_theta = shard_theta(dataset.root_theta, config)

    if config.hp.expert_count:
        if config.hp.model_arch == "grok":
            model = PagedGrokModelV1(dataset.root_theta, config)
        else:
            model = PagedMixtralModelV1(dataset.root_theta, config)
    else:
        model = PagedLlamaModelV1(dataset.root_theta, config)

    if args.save_intermediates_path:
        from ..utils.patching import SaveModuleResultTensorsPatch

        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(model)
    generator = TorchGenerator(model, tokenizer, dump_bins=args.dump_bins)

    print(f":: Prompting:")
    for prompt in prompts:
        print(f"    {prompt.encode()}")

    batch = generator.begin_batch(prompts)
    print(f":: Prompt tokens: {batch.token_ids}")
    batch.prefill()
    print(batch.detokenize())

    if args.save_intermediates_path:
        intermediates_saver.save_file(
            args.save_intermediates_path + "_prefill.safetensors"
        )
    if not args.skip_decode:
        counter = 0
        while not batch.done:
            batch.decode()
            if args.save_intermediates_path:
                intermediates_saver.save_file(
                    args.save_intermediates_path + f"_step_{counter}.safetensors"
                )
            print(f":: Result tokens: {batch.results}")
            batch.print_current_results()
            counter += 1


if __name__ == "__main__":
    main()
