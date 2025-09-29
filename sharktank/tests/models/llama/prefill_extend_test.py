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
    def test_prefill_vs_extend_prefill(self):
        seed = 0
        torch.manual_seed(seed)

        batch_size = 2
        theta, config = generate(seed)
        config.block_seq_stride = 32
        model = PagedLlmModelV1(theta, config)

        prefill_args = make_random_prefill_args(model, batch_size)
        start_positions = None
        # logits_prefill = model.prefill(**prefill_args, start_positions=start_positions)

        config.attention_kernel = "wave"
        # set up prefill_extend_args from prefill_args
        prefill_tokens = prefill_args["tokens"]
        prefill_seq_lens = prefill_args["seq_lens"]
        prefill_seq_block_ids = prefill_args["seq_block_ids"]
        prefill_cache_state = prefill_args["cache_state"]
        # chunk tokens and seq_lens by chunk size
        flattened_tokens_no_pad = torch.cat([seq[seq != 0] for seq in prefill_tokens])
        flattened_tokens_no_pad = flattened_tokens_no_pad.unsqueeze(0)
        prefill_extend_logits = []
        bs = prefill_seq_lens.shape[0]
        offsets = torch.cumsum(
            torch.cat(
                [torch.tensor([0], device=prefill_seq_lens.device), prefill_seq_lens]
            ),
            dim=0,
        )
        for i in range(bs):
            start, end = offsets[i].item(), offsets[i + 1].item()
            flattened_tokens_i = flattened_tokens_no_pad[:, start:end]
            start_positions = None
            logits_i = model.prefill_extend(
                flattened_tokens_i,
                seq_lens=prefill_seq_lens[i],
                start_positions=start_positions,
                seq_block_ids=prefill_seq_block_ids[i : i + 1, :],
                cache_state=prefill_cache_state,
            )
            prefill_extend_logits.append(logits_i)
        breakpoint()


if __name__ == "__main__":
    unittest.main()
