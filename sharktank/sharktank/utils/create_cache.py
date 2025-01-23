# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..layers import *


def create_paged_kv_cache(config: LlamaModelConfig) -> PagedKVCache:
    if config.kv_cache_type != "paged":
        raise ValueError("Model does not use paged kv cache, cannot create kv cache")

    hp = config.hp
    return PagedKVCache(
        transformer_block_count=hp.block_count,
        attn_head_count=hp.attention_head_count_kv,
        attn_head_dim=hp.attn_head_dim,
        block_seq_stride=config.block_seq_stride,
        device=config.device,
        dtype=config.attention_dtype,
        shard_count=config.tensor_parallelism_size,
    )
