# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .testing import make_random_llama_theta

from sharktank.layers.configs import LlamaHParams
from sharktank.models.llama.llama import LlamaModelConfig
from sharktank.types import Dataset

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_llama.irpa")


def generate(seed):
    torch.manual_seed(seed)
    dtype = torch.float16
    block_seq_stride = 16
    max_blocks = 8
    attention_head_count = 8
    attn_head_dim = 32
    attention_head_count_kv = 4
    rope_dimension_count = 32
    vocabulary_size = 256

    config = LlamaModelConfig(
        hp=LlamaHParams(
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=3,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=500000.0,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=0.01,
            attention_head_count_kv=attention_head_count_kv,
            expert_count=0,
            expert_used_count=0,
            model_arch="llama",
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        attention_dtype=dtype,
    )

    theta = make_random_llama_theta(
        config=config,
        vocab_size=vocabulary_size,
    )
    return theta, config


def main():
    args = parser.parse_args()
    theta, config = generate(args.seed)

    config_dict = config.hp.to_gguf_props()

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)


if __name__ == "__main__":
    main()
