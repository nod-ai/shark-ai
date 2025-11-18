# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from pathlib import Path
import collections
import math
import numpy as np

import torch

from amdsharktank.layers.kv_cache import CacheAllocation
from amdsharktank.types import *
from amdsharktank.models.llm import PagedLlmModelV1

from amdsharktank.ops import replicate, unshard
from amdsharktank.utils.attention import *
from amdsharktank.utils.debugging import trace_tensor
from amdsharktank.utils.tokenizer import InferenceTokenizer
from amdsharktank.utils.evaluate import *
from amdsharktank.utils.logging import get_logger

logger = get_logger(__name__)


class TorchGenerator:
    """Generator that runs directly on the Torch model."""

    def __init__(
        self,
        model: PagedLlmModelV1,
        tokenizer: Optional[InferenceTokenizer] = None,
        # Need to look at the model more for this.
        end_token: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = end_token

    @property
    def block_seq_stride(self) -> int:
        return self.model.config.block_seq_stride

    def preprocess_prompts(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.tokenizer.encode(texts=prompts, add_start_token=False)[0]

        logger.info(f":: Prompt tokens:")
        for idx, prompt in enumerate(prompts):
            logger.info(
                f"    prompt_{idx}: \n    {prompt.encode()} \n    {token_ids[idx]}\n"
            )

        token_ids, seq_lens = pad_tokens(
            token_ids,
            pad_to_multiple_of=self.model.config.block_seq_stride,
            device=self.model.device,
        )

        return token_ids, seq_lens

    def generate_random_tokens(
        self, batch_size: int, prompt_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.randint(
            low=0,
            high=self.model.config.hp.vocab_size,
            size=[batch_size, prompt_seq_len],
        )
        # TODO: refactor to not use list[list[int]] as this is not efficient.
        token_ids = [[int(t) for t in s] for s in token_ids]
        token_ids, seq_lens = pad_tokens(
            token_ids, pad_to_multiple_of=self.model.config.block_seq_stride
        )
        token_ids = torch.tensor(token_ids, device=self.model.device)
        seq_lens = torch.tensor(seq_lens, device=self.model.device)
        return token_ids, seq_lens

    def begin_batch(
        self,
        token_ids: torch.tensor,
        seq_lens: torch.tensor,
        page_cache_size: int = None,
        dump_path: Path = None,
        dump_decode_steps: int = 1,
        max_decode_steps: int = 20,
    ) -> "Batch":
        bs = token_ids.shape[0]

        self.page_cache_size = (
            page_cache_size
            or ((token_ids[0].shape[0] // self.model.config.block_seq_stride) * bs + 1)
            * 2
        )

        cache_state = self.model.cache.allocate(self.page_cache_size)
        self.free_pages = list(range(1, self.page_cache_size))

        assert (
            max_decode_steps >= dump_decode_steps
        ), f"Expected max_decode_steps >= dump_decode_steps but got max_decode_steps={max_decode_steps}, dump_decode_steps={dump_decode_steps}"

        return Batch(
            self,
            token_ids=token_ids,
            seq_lens=seq_lens,
            cache_state=cache_state,
            bs=bs,
            dump_path=dump_path,
            dump_decode_steps=dump_decode_steps,
            max_decode_steps=max_decode_steps,
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
        cache_state: CacheAllocation,
        bs: int,
        dump_path: Path,
        dump_decode_steps: int,
        max_decode_steps: int,
    ):
        self.bs = bs
        assert seq_lens.shape[0] == self.bs
        self.parent = parent
        self.token_ids = token_ids
        self.seq_lens = seq_lens
        # TODO: This doesn't appear to handle PP models properly
        self.cache_state = cache_state
        self.results: list[list[int]] = [[] for _ in range(self.bs)]
        self.done_result_indices: set[int] = set()
        self.dump_path = dump_path
        self.dump_decode_steps = dump_decode_steps
        self.decode_step = 0
        self.max_decode_steps = max_decode_steps

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
            len(self.done_result_indices) == self.bs
            or len(self.parent.free_pages) == 0
            or (self.max_decode_steps <= self.decode_step)
        )

    def detokenize(self) -> list[str]:
        return self.parent.tokenizer.decode(self.results)

    def print_current_results(self):
        if len(self.results[0]) == 1:
            phase = "prefill"
        else:
            phase = "decode"

        logger.info(f":: {phase} result tokens:")
        results = self.detokenize()
        for i, s in enumerate(results):
            seq_len = int(self.seq_lens[i])
            logger.info(
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

    def dump_args(
        self,
        phase: str,
        arg_name: str,
        arg: AnyTensor | list[AnyTensor],
        decode_step: int | None = None,
        rank: int | None = None,
    ):
        if isinstance(arg, collections.abc.Sequence):
            for i, a in enumerate(arg):
                subarg_name = arg_name if len(arg) == 1 else f"{arg_name}_{i}"
                self.dump_args(phase, subarg_name, a, decode_step, rank)
        elif isinstance(arg, ShardedTensor):
            for rank in range(arg.shard_count):
                self.dump_args(
                    phase, arg_name, arg.shards[rank]._data, decode_step, rank
                )
        else:
            arg = unbox_tensor(arg)
            if arg.dtype in [torch.float8_e4m3fnuz, torch.bfloat16]:
                arg = arg.to(torch.uint8)
            if phase == "decode":
                arg_name = f"{arg_name}_{decode_step}"
            rank_str = "" if (rank is None) else f".rank{rank}"

            file_path = Path(self.dump_path, f"{phase}_{arg_name}{rank_str}.npy")
            np.save(file_path, arg.cpu().numpy())

    def prefill(self):
        model = self.parent.model
        seq_block_ids = self.pad_block_ids()
        token_ids = self.token_ids
        trace_tensor("prefill.token_ids", self.token_ids)
        trace_tensor("prefill.seq_block_ids", seq_block_ids)

        shard_count = model.config.tensor_parallelism_size
        num_pipelines = model.config.pipeline_parallelism_size

        assert shard_count == 1
        assert num_pipelines == 1

        if self.dump_path is not None:
            logger.info(f"\nSaving prefill args to {Path(self.dump_path)}\n")

            self.dump_args(phase="prefill", arg_name="token_ids", arg=token_ids)
            self.dump_args(phase="prefill", arg_name="seq_lens", arg=self.seq_lens)
            self.dump_args(phase="prefill", arg_name="seq_block_ids", arg=seq_block_ids)
            self.dump_args(
                phase="prefill", arg_name="cache_state", arg=self.cache_state.allocation
            )

        self.prefill_logits = model.prefill(
            token_ids,
            seq_lens=self.seq_lens,
            seq_block_ids=seq_block_ids,
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
        seq_block_ids = self.pad_block_ids()

        trace_tensor("decode.token_ids", token_batch)
        trace_tensor("decode.start_positions", start_positions)
        trace_tensor("decode.seq_block_ids", seq_block_ids)

        shard_count = model.config.tensor_parallelism_size
        num_pipelines = model.config.pipeline_parallelism_size

        assert shard_count == 1
        assert num_pipelines == 1

        if self.dump_path is not None and self.decode_step < self.dump_decode_steps:
            print(f"\nSaving decode args to {Path(self.dump_path)}\n")

            self.dump_args(
                phase="decode",
                arg_name="next_tokens",
                arg=token_batch,
                decode_step=self.decode_step,
            )
            self.dump_args(
                phase="decode",
                arg_name="start_positions",
                arg=start_positions,
                decode_step=self.decode_step,
            )
            self.dump_args(
                phase="decode",
                arg_name="seq_lens",
                arg=self.seq_lens,
                decode_step=self.decode_step,
            )
            self.dump_args(
                phase="decode",
                arg_name="seq_block_ids",
                arg=seq_block_ids,
                decode_step=self.decode_step,
            )
            self.dump_args(
                phase="decode",
                arg_name="cache_state",
                arg=self.cache_state.allocation,
                decode_step=self.decode_step,
            )

        self.decode_logits = model.decode(
            token_batch,
            seq_lens=self.seq_lens,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
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
        self.decode_step += 1
        return tokens

    def pad_block_ids(self) -> torch.Tensor:
        max_length = max(len(r) for r in self.seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in self.seq_block_ids]
        return torch.tensor(rows, device=self.parent.model.device)
