# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig

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


st_rotary = RotaryEmbeddingLayer(
    rope_dimension_count=dims, max_seqlen=2048, rope_freq_base=500000, use_hf=True
)

hf_rotary = HFRotaryEmbedding()

example = torch.rand(bs, length, heads, dims, dtype=torch.bfloat16)
positions = torch.arange(0, length)[None, :].repeat(bs, 1)

mask = st_rotary.compute_batch_mask(
    start_positions=torch.arange(0, bs), batch_seq_len=1
)
st_results = st_rotary.apply_batched_mask_unsharded(xt=example, mask=mask).squeeze(0)
hf_results = hf_rotary.forward(xt=example, positions=torch.arange(0, bs).unsqueeze(1))
print(st_results.shape)
st_results = st_results.transpose(1, 2, 0)
print(hf_results.shape)
print(torch.max(torch.abs(hf_results - st_results)))

hf_results = hf_rotary(xt=example, positions=positions)
st_results = st_rotary.forward(xt=example, start_index=0)

print(torch.max(torch.abs(hf_results - st_results)))
# for name, hf_result, st_result in zip(["xt","cos", "sin"], hf_results, st_results):
#    print(name)
#    print(hf_result.shape)
#    print(hf_result.dtype)
#    print(st_result.shape)
#    print(st_result.dtype)
#    print(torch.sum(hf_result))

#    print(torch.max(torch.abs(hf_result - st_result)))

# if __name__ == "__main__":
#    unittest.main()
