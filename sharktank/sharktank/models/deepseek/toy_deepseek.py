# Copyright 2025 Advanced Micro Devices, Inc.
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

from dataclasses import asdict

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_deepseek.irpa")


def generate(seed):
    torch.manual_seed(seed=12345)
    dtype = torch.float32
    block_seq_stride = 16
    max_blocks = 8

    attn_head_dim = 16
    rope_dimension_count = 16
    vocabulary_size = 256
    expert_count = 4
    used_experts = 2

    config = LlamaModelConfig(
        hp=LlamaHParams(
            context_length=block_seq_stride * max_blocks,
            embedding_length=32,
            block_count=1,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=500000.0,
            attention_head_count=4,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=0.01,
            attention_head_count_kv=4,
            nope_dim=32,
            kv_latent_dim=16,
            v_head_dim=32,
            expert_count=expert_count,
            expert_used_count=used_experts,
            model_arch="deepseek",
            expert_score_func="sigmoid",
            route_scale=2.0,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        attention_dtype=dtype,
    )

    theta = make_random_deepseek_theta(
        config=config,
        vocab_size=vocabulary_size,
    )
    return theta, config


def main():
    args = parser.parse_args()
    theta, config = generate(args.seed)

    flat = theta.flatten()
    for k in sorted(flat.keys()):
        print(f"{k:<50} {str(flat[k].shape):<20} {str(flat[k].dtype):<10}")

    config_dict = {
        "hparams": asdict(config.hp),
    }

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)


if __name__ == "__main__":
    main()
