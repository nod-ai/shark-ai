# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

import torch

from sharktank.models.llama.toy_llama import generate
from sharktank.models.llm.llm import PagedLlmModelV1
from sharktank.models.llm.testing import make_random_prefill_args


class TestPrefillExtend:
    def test_single_request(self):
        seed = 0
        torch.manual_seed(seed)

        batch_size = 1
        chunk_size = 128
        theta, config = generate(seed)
        config.block_seq_stride = 32
        model = PagedLlmModelV1(theta, config)

        # prefill_args = make_random_prefill_args(model, batch_size)
        seq_lens = torch.tensor([256])
        max_seq_len = torch.max(seq_lens)
        token_ids = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.int32,
            device=model.device,
        )

        for b in range(batch_size):
            token_ids[b, : seq_lens[b]] = torch.randint(
                low=0,
                high=model.config.hp.vocab_size,
                size=(seq_lens[b].item(),),
                dtype=torch.int32,
                device=model.device,
            )

        seq_block_ids = torch.arange(
            batch_size * max_seq_len // model.config.block_seq_stride,
            device=model.device,
        ).view(batch_size, -1)
        cache_state = model.cache.allocate(
            page_count=seq_block_ids[0].numel() + batch_size
        )
        # cache_state = [torch.rand_like(cache_state[0])]
        start_positions = None
        prefill_logits = model.prefill(
            token_ids,
            seq_lens=seq_lens,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )

        config.attention_kernel = "wave"
        model = PagedLlmModelV1(theta, config)
        # chunk tokens and seq_lens by chunk size
        flattened_tokens_no_pad = torch.cat([token[token != 0] for token in token_ids])
        flattened_tokens_no_pad = flattened_tokens_no_pad.unsqueeze(0)
        prefill_extend_logits = []
        bs = seq_lens.shape[0]
        num_chunks = torch.tensor(
            seq_lens // chunk_size, dtype=torch.int32, device=model.device
        )
        # Remainder for the last chunk
        last_chunk = seq_lens % chunk_size
        last_chunk = torch.tensor(
            seq_lens % chunk_size, dtype=torch.int32, device=model.device
        )
        chunk_sizes = torch.tensor(
            [chunk_size] * num_chunks + ([last_chunk] if last_chunk > 0 else [])
        )
        offsets = torch.cumsum(
            torch.cat([torch.tensor([0], device=model.device), chunk_sizes]),
            dim=0,
        )
        for i in range(len(chunk_sizes)):
            start, end = offsets[i].item(), offsets[i + 1].item()
            flattened_tokens_i = flattened_tokens_no_pad[:, start:end]
            cur_seq_lens = (start + chunk_sizes[i]).unsqueeze(0)
            start_positions = torch.tensor([start], device=model.device)
            seq_lens = cur_seq_lens
            logits_i = model.prefill_extend(
                flattened_tokens_i,
                seq_lens=cur_seq_lens,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )
            prefill_extend_logits.append(logits_i)
        final_prefill_extend_logits = torch.cat(prefill_extend_logits, dim=1)
        torch.testing.assert_close(
            final_prefill_extend_logits, prefill_logits, rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
