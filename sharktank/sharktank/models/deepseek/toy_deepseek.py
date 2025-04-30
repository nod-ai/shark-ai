# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.models.deepseek.testing import make_random_deepseek_theta

from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
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
    block_seq_stride = 32
    max_blocks = 8

    rope_dimension_count = 64
    vocabulary_size = 256
    expert_count = 4
    used_experts = 2
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    attn_head_dim = qk_nope_head_dim + qk_rope_head_dim

    config = LlamaModelConfig(
        hp=LlamaHParams(
            model_arch="deepseek2",
            context_length=block_seq_stride * max_blocks,
            embedding_length=32,
            block_count=4,
            feed_forward_length=23,
            attention_head_count=4,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=9.0,
            attention_head_count_kv=4,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=128,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=10000.0,
            expert_count=expert_count,
            expert_used_count=used_experts,
            expert_shared_count=1,
            n_expert_groups=8,
            n_limited_groups=4,
            n_dense_layers=3,
            route_scale=2.5,
            rope_scaling_type="yarn",
            rope_scaling_factor=40.0,
            rope_scaling_original_context_length=4096,
            rope_scaling_yarn_log_multiplier=0.10000000149011612,
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

    config_dict = config.hp.to_gguf_props()

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)


if __name__ == "__main__":
    main()
