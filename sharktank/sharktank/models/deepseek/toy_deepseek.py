# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .testing import make_random_deepseek_theta

from sharktank.layers.configs import LlamaHParams
from sharktank.models.llama.llama import LlamaModelConfig
from sharktank.types import Dataset

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_deepseek.irpa")


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float32
    block_seq_stride = 16
    max_blocks = 8
    attention_head_count = 8  # num_attention_heads=128
    attn_head_dim = (
        16  # qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim 128+64=192
    )
    attention_head_count_kv = 2  # num_key_value_heads=128
    rope_dimension_count = 16  # qk_rope_head_dim=64
    vocabulary_size = 129280
    expert_count = 16
    used_experts = 8

    config = LlamaModelConfig(
        hp=LlamaHParams(
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=1,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=10000.0,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=1e-06,
            attention_head_count_kv=attention_head_count_kv,
            expert_count=expert_count,
            expert_used_count=used_experts,
            model_arch="grok",
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        attention_dtype=dtype,
    )

    theta = make_random_deepseek_theta(
        config=config,
        vocab_size=vocabulary_size,
    )

    config_dict = config.hp.to_gguf_props()

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)


if __name__ == "__main__":
    main()
