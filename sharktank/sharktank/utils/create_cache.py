# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers import *
from sharktank.types.quantizers import StaticScaledQuantizer


def create_paged_attention(
    config: "LlamaModelConfig",
    use_rope: bool,
    block_index: int,
    k_quantizer: StaticScaledQuantizer | None = None,
    v_quantizer: StaticScaledQuantizer | None = None,
) -> PagedAttention:
    # TODO: Add deepseek PagedLatentAttention

    if config.kv_cache_type != "paged":
        raise ValueError("Model does not use paged kv cache, cannot create kv cache")

    hp = config.hp
    dtype = config.kv_cache_dtype or config.attention_dtype
    return PagedAttention(
        transformer_block_count=hp.block_count,
        attention_chunk_size=config.attention_chunk_size,
        transformer_block_index=block_index,
        attn_head_count=hp.attention_head_count_kv,
        attn_head_dim=hp.attn_head_dim,
        attn_type=attn_type_map[hp.model_arch],
        cache_partition_count=2,  # One for each of K/V.
        block_seq_stride=config.block_seq_stride,
        device=config.device,
        use_rope=use_rope,
        cache_dtype=dtype,
        attn_dtype=config.attention_dtype,
        activation_dtype=config.activation_dtype,
        k_quantizer=k_quantizer,
        v_quantizer=v_quantizer,
    )
