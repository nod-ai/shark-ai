# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import numpy as np
import torch

from sharktank.layers import *
from sharktank.types import *
from sharktank.models.llm import *
from sharktank.models.llama.tools.data_utils import write_ndarray_to_bin

from sharktank.ops import replicate, unshard
from sharktank.utils.debugging import trace_tensor
from sharktank.utils.tokenizer import InferenceTokenizer


class TorchGenerator:
    """Generator that runs directly on the Torch model."""

    def __init__(
        self,
        model: PagedLlmModelV1,
        tokenizer: InferenceTokenizer,
        # Need to look at the model more for this.
        end_token: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = end_token

    @property
    def block_seq_stride(self) -> int:
        return self.model.cache.block_seq_stride

    def preprocess_prompts(
        self,
        prompts: list[str],
    ):
        token_ids = self.tokenizer._encode(texts=prompts, add_start_token=False)

        print(f":: Prompt tokens:")
        for idx, prompt in enumerate(prompts):
            print(f"    prompt_{idx}: \n    {prompt.encode()} \n    {token_ids[idx]}\n")

        token_ids, seq_lens = self.tokenizer.pad_tokens(
            token_ids, pad_to_multiple_of=self.model.cache.pad_sequence_stride
        )

        token_ids = torch.tensor(token_ids, device=self.model.device)
        seq_lens = torch.tensor(seq_lens, device=self.model.device)

        return token_ids, seq_lens

    def begin_batch(
        self,
        token_ids: torch.tensor,
        seq_lens: torch.tensor,
        page_cache_size: int = None,
        dump_bins: bool = False,
    ):
        bs = token_ids.shape[0]

        self.page_cache_size = (
            page_cache_size
            or ((token_ids[0].shape[0] // self.model.config.block_seq_stride) * bs + 1)
            * 2
        )

        cache_state = self.model.cache.allocate(self.page_cache_size)
        self.free_pages = list(range(1, self.page_cache_size))

        return Batch(
            self,
            token_ids=token_ids,
            seq_lens=seq_lens,
            cache_state=cache_state,
            bs=bs,
            dump_bins=dump_bins,
        )

    def alloc_page(self) -> int:
        return self.free_pages.pop()

    def release_page(self, index: int):
        self.free_pages.append(index)


class Batch:
    def __init__(
        self,
        parent: TorchGenerator,
        token_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        cache_state: list[torch.Tensor],
        bs: int,
        dump_bins: bool = False,
    ):
        self.bs = bs
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
        return (
            len(self.done_result_indices) == self.bs or len(self.parent.free_pages) == 0
        )

    def detokenize(self) -> list[str]:
        return self.parent.tokenizer.decode(self.results)

    def print_current_results(self):
        if len(self.results[0]) == 1:
            phase = "prefill"
        else:
            phase = "decode"

        print(f":: {phase} result tokens:")
        results = self.detokenize()
        for i, s in enumerate(results):
            seq_len = int(self.seq_lens[i])
            print(
                f"   prompt_{i}({len(self.results[i])}, {seq_len}): {s} \n   {self.results[i]}"
            )

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
                if self.done:
                    break
                block_ids_row.append(self.parent.alloc_page())

    def prefill(self):
        model = self.parent.model
        attention_mask = model.attention_mask(
            model.input_mask(self.seq_lens, self.token_ids.shape[1])
        )
        seq_block_ids_tensor = self.pad_block_ids()
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

        self.prefill_logits = model.prefill(
            token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        self.prefill_logits = unshard(self.prefill_logits)

        # TODO: Generalize the sampling and don't make it swap on/off cpu.
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(self.prefill_logits, self.seq_lens)
        ).unsqueeze(1)
        self.add_result_token(tokens)
        return tokens.to(device=model.device)

    def decode(self, token_batch=None):

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
        trace_tensor("decode.token_ids", token_batch)
        trace_tensor("decode.start_positions", start_positions)
        trace_tensor("decode.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("decode.attention_mask", decode_attention_mask)

        if model.config.tensor_parallelism_size != 1:
            tp = model.config.tensor_parallelism_size
            token_batch = replicate(token_batch, tp)
            start_positions = replicate(start_positions, tp)
            seq_block_ids_tensor = replicate(seq_block_ids_tensor, tp)
            decode_attention_mask = replicate(decode_attention_mask, tp)

        if self.dump_bins:
            write_ndarray_to_bin(
                token_batch.numpy(),
                f"decode_next_tokens_{'x'.join([str(x)for x in token_batch.shape])}xi64.bin",
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
                torch.tensor(token_batch.shape[0]).to(torch.int64).numpy(),
                f"decode_seq_lens_1xi64.bin",
            )
            write_ndarray_to_bin(
                self.cache_state[0].to(torch.float8_e4m3fnuz).to(torch.uint8).numpy(),
                f"decode_cache_state_{'x'.join([str(x) for x in self.cache_state[0].shape])}xf8E4M3FNUZ.bin",
            )

        self.decode_logits = model.decode(
            token_batch,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        self.decode_logits = unshard(self.decode_logits)

        trace_tensor("decode.logits", self.decode_logits)
        # # TODO: Normalize the output of extract_tokens_from_logits into
        # # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(self.decode_logits, [1] * self.bs),
            device=self.parent.model.device,
        ).unsqueeze(1)
        self.add_result_token(tokens)
        return tokens

    def pad_block_ids(self) -> torch.Tensor:
        max_length = max(len(r) for r in self.seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in self.seq_block_ids]
        return torch.tensor(rows, device=self.parent.model.device)
