# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.layers import *
from amdsharktank.types.quantizers import StaticScaledQuantizer


def create_paged_attention(
    config: "LlamaModelConfig",
    kv_cache: KVCache,
    use_rope: bool,
    block_index: int,
    k_quantizer: StaticScaledQuantizer | None = None,
    v_quantizer: StaticScaledQuantizer | None = None,
) -> PagedAttention:
    if config.kv_cache_type != "paged":
        raise ValueError("Model does not use paged kv cache, cannot create kv cache")

    attn_type = attn_type_map[config.hp.model_arch]

    attention_class_map = {
        "gqa": PagedGQAttention,
        "mla": PagedMLAttention,
    }

    attention_class = attention_class_map.get(attn_type)
    if attention_class is None:
        error_msg = f"Unsupported attention type to create PagedAttention: {attn_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return attention_class(
        attention_chunk_size=config.attention_chunk_size,
        transformer_block_index=block_index,
        kv_cache=kv_cache,
        use_rope=use_rope,
        attn_dtype=config.attention_dtype,
        activation_dtype=config.activation_dtype,
        k_quantizer=k_quantizer,
        v_quantizer=v_quantizer,
    )
