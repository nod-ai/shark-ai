# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import torch

from sharktank.layers import *
from sharktank.types import *
from sharktank.models.llm import *

from ..utils.debugging import trace_tensor
from ..utils.tokenizer import InferenceTokenizer


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

    def begin_batch(
        self, prompts: list[str], add_start_token: bool, page_cache_size: int = 128
    ):
        token_ids, seq_lens = self.tokenizer.encode(
            prompts,
            pad_to_multiple_of=self.model.cache.pad_sequence_stride,
            add_start_token=add_start_token,
        )
        token_ids = torch.tensor(token_ids, device=self.model.device)
        seq_lens = torch.tensor(seq_lens, device=self.model.device)

        if self.model.config.kv_cache_type == "paged":
            cache_state = self.model.cache.allocate(page_cache_size)
            self.free_pages = list(range(1, page_cache_size))
        return Batch(self, token_ids, seq_lens, cache_state)

    def begin_eval_batch(
        self,
        token_batch: torch.tensor,
        seq_lens_batch: torch.tensor,
        bs: int,
        page_cache_size: int = 128,
    ):
        if self.model.config.kv_cache_type == "paged":
            cache_state = self.model.cache.allocate(page_cache_size)
            self.free_pages = list(range(1, page_cache_size))
        return Batch(self, token_batch, seq_lens_batch, cache_state)

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
    ):
        self.bs = token_ids.shape[0]
        # assert seq_lens.shape[0] == self.bs
        self.parent = parent
        self.token_ids = token_ids
        self.seq_lens = seq_lens
        self.cache_state = cache_state
        self.results: list[list[int]] = [[] for _ in range(self.bs)]
        self.done_result_indices: set[int] = set()

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
        trace_tensor("prefill.token_ids", self.token_ids)
        trace_tensor("prefill.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("prefill.attention_mask", attention_mask)
        self.prefill_logits = model.prefill(
            self.token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        # TODO: Generalize the sampling and don't make it swap on/off cpu.
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(self.prefill_logits, self.seq_lens)
        ).unsqueeze(1)
        self.add_result_token(tokens)
        self.next_tokens = tokens.to(device=model.device)

    def decode(self, token_batch):
        self.token_batch = token_batch

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
        trace_tensor("decode.token_ids", self.token_ids)
        trace_tensor("decode.start_positions", start_positions)
        trace_tensor("decode.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("decode.attention_mask", decode_attention_mask)

        self.decode_logits = model.decode(
            self.token_batch,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        trace_tensor("decode.logits", self.decode_logits)
        # # TODO: Normalize the output of extract_tokens_from_logits into
        # # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(self.decode_logits, [1] * self.bs),
            device=self.parent.model.device,
        ).unsqueeze(1)
        self.add_result_token(tokens)
        self.next_tokens = tokens

    def pad_block_ids(self) -> torch.Tensor:
        max_length = max(len(r) for r in self.seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in self.seq_block_ids]
        return torch.tensor(rows, device=self.parent.model.device)
