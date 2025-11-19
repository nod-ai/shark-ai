# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .testing import make_random_llama_theta

from amdsharktank.layers.configs import (
    LlamaHParams,
    LlamaModelConfig,
    ParallelismConfig,
)
from amdsharktank.types import Dataset

import argparse
import math
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/toy_llama.irpa")


def generate(
    seed,
    dtype_rest: torch.dtype = torch.float16,
    dtype_norm: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    block_seq_stride = 16
    max_blocks = 8
    attention_head_count = 8
    attn_head_dim = 32
    attention_head_count_kv = 4
    rope_dimension_count = 32
    vocabulary_size = 256

    config = LlamaModelConfig(
        hp=LlamaHParams(
            vocab_size=vocabulary_size,
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=3,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=500000.0,
            rope_interleave_emb=True,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=0.01,
            attention_head_count_kv=attention_head_count_kv,
            expert_count=0,
            expert_used_count=0,
            model_arch="llama",
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype_rest,
        attention_dtype=dtype_rest,
    )

    theta = make_random_llama_theta(
        config=config,
        vocab_size=vocabulary_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
    )
    return theta, config


def make_config2(
    quantization_block_size: int = 1,
    pipeline_parallelism_size: int = 1,
    tensor_parallelism_size: int = 1,
    dtype_rest: torch.dtype = torch.float16,
    dtype_norm: torch.dtype = torch.float32,
) -> LlamaModelConfig:
    block_seq_stride = 3
    max_blocks = 4
    feed_forward_length = 4
    attention_head_count_kv = 5
    block_count = 2  # number of layers

    # Make sizes respect quantization block boundary and tensor sharding boundary.
    # This is required because quantization and tensor-sharding can't operate on
    # uneven blocks/splits.
    feed_forward_length = math.lcm(feed_forward_length, quantization_block_size)
    feed_forward_length = math.lcm(feed_forward_length, tensor_parallelism_size)
    feed_forward_length = math.lcm(
        feed_forward_length, quantization_block_size * tensor_parallelism_size
    )
    attention_head_count_kv = math.lcm(attention_head_count_kv, quantization_block_size)
    attention_head_count_kv = math.lcm(attention_head_count_kv, tensor_parallelism_size)

    # We need at least 1 block per pipeline stage.
    block_count = max(block_count, pipeline_parallelism_size)

    attention_head_count = attention_head_count_kv * 2
    rope_dimension_count = 4 * 2
    attn_head_dim = rope_dimension_count
    vocabulary_size = 9

    return LlamaModelConfig(
        hp=LlamaHParams(
            vocab_size=vocabulary_size,
            context_length=block_seq_stride * max_blocks,
            embedding_length=attention_head_count * attn_head_dim,
            block_count=block_count,
            feed_forward_length=feed_forward_length,
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
        parallelism_config=ParallelismConfig.default_config(
            block_count=block_count,
            pp=pipeline_parallelism_size,
            tp=tensor_parallelism_size,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype_rest,
        attention_dtype=dtype_rest,
        fake_quant=False,
    )


def generate2(
    seed,
    quantization_block_size: int = 1,
    pipeline_parallelism_size: int = 1,
    tensor_parallelism_size: int = 1,
    dtype_rest: torch.dtype = torch.float16,
    dtype_norm: torch.dtype = torch.float32,
):
    """TODO: reconcile with other toy model that is really not toy as it has
    vocabulary size of 256, attn_head_dim of 32, etc."""
    torch.manual_seed(seed)
    config = make_config2(
        quantization_block_size=quantization_block_size,
        pipeline_parallelism_size=pipeline_parallelism_size,
        tensor_parallelism_size=tensor_parallelism_size,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
    )

    theta = make_random_llama_theta(
        config=config,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
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
