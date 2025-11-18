# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from amdsharktank.layers import build_rotary_layer
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig
import unittest


class HFRotaryComparisonTest(unittest.TestCase):
    def test(self):
        test_dtype = torch.bfloat16
        bs = 2
        length = 5
        heads = 3
        dims = 128
        rope_scaling = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        hf_config = LlamaConfig(
            rope_scaling=rope_scaling,
            max_position_embeddings=131072,
            rope_theta=500000,
        )
        torch.manual_seed(123456)

        class HFRotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._rotary = LlamaRotaryEmbedding(config=hf_config)

            def forward(self, *, xt, positions):
                cos, sin = self._rotary(xt, positions)
                xt = xt.transpose(1, 2)
                return apply_rotary_pos_emb(xt, xt, cos, sin)[0].transpose(1, 2)

        st_rotary = build_rotary_layer(
            rope_dimension_count=dims,
            rope_freq_base=500000,
            interleave=False,
            dtype=test_dtype,
            yarn_beta_slow=1,
            yarn_beta_fast=4,
            yarn_factor=8,
            yarn_original_context_len=8192,
        )

        hf_rotary = HFRotaryEmbedding()

        example = torch.rand(bs, length, heads, dims, dtype=test_dtype)
        positions = torch.arange(0, length)[None, :].repeat(bs, 1)

        decode_example = torch.rand(bs, 1, heads, dims, dtype=test_dtype)
        mask = st_rotary.compute_batch_mask(
            start_positions=torch.arange(0, bs), batch_seq_len=1
        )
        st_results = st_rotary.apply_batched_mask(xt=decode_example, mask=mask)
        hf_results = hf_rotary.forward(
            xt=decode_example, positions=torch.arange(0, bs).unsqueeze(1)
        )
        torch.testing.assert_close(st_results, hf_results)

        hf_results = hf_rotary(xt=example, positions=positions)
        st_results = st_rotary.forward(xt=example)
        torch.testing.assert_close(st_results, hf_results)


if __name__ == "__main__":
    unittest.main()
