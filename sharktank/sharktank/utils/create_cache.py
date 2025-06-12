# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers import *


def create_paged_kv_cache(config: LlamaModelConfig) -> PagedAttention:
    if config.kv_cache_type != "paged":
        raise ValueError("Model does not use paged kv cache, cannot create kv cache")

    attn_type_map = {
        "llama": "gqa",
        "grok": "gqa",
        "deepseek2": "mla",
        "llama4": "gqa",
    }

    hp = config.hp
    dtype = config.kv_cache_dtype or config.attention_dtype
    return PagedAttention(
        transformer_block_count=hp.block_count,
        block_to_pipeline_map=config.block_to_pipeline_map,
        pipeline_to_device_map=config.pipeline_to_device_map,
        attn_head_count=hp.attention_head_count_kv,
        attn_head_dim=hp.attn_head_dim,
        attn_type=attn_type_map[hp.model_arch],
        cache_partition_count=2,  # One for each of K/V.
        block_seq_stride=config.block_seq_stride,
        device=config.device,
        cache_dtype=dtype,
        attn_dtype=config.attention_dtype,
        shard_count=config.tensor_parallelism_size,
    )
