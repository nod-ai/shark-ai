# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.models.deepseek.testing import make_random_deepseek_theta

from amdsharktank.layers.configs import LlamaHParams, LlamaModelConfig
from amdsharktank.types import Dataset, Theta

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_deepseek.irpa")


def generate(
    seed: int,
    dtype_rest: torch.dtype = torch.float16,
    dtype_norm: torch.dtype = torch.float32,
) -> tuple[Theta, LlamaModelConfig]:
    torch.manual_seed(seed=seed)

    # Constants
    rope_dimension_count = 64
    block_seq_stride = 32

    max_blocks = 8
    vocabulary_size = 256
    expert_count = 4
    used_experts = 2
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    attn_head_dim = qk_nope_head_dim + qk_rope_head_dim

    config = LlamaModelConfig(
        hp=LlamaHParams(
            model_arch="deepseek2",
            vocab_size=vocabulary_size,
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
            rope_interleave_emb=True,
            expert_count=expert_count,
            expert_used_count=used_experts,
            expert_shared_count=1,
            moe_intermediate_size=7,
            n_expert_groups=2,
            n_limited_groups=2,
            n_dense_layers=3,
            route_scale=2.5,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype_norm,
        attention_dtype=dtype_norm,
        kv_cache_dtype=dtype_norm,
    )

    theta = make_random_deepseek_theta(
        config=config,
        vocab_size=vocabulary_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
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
