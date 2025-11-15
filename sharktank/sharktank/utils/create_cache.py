# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers import *
from sharktank.types.quantizers import StaticScaledQuantizer
from typing import Optional


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
    use_extend_attention = config.use_extend_attention
    attention_class = select_attention_class(attn_type, use_extend_attention)

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


def select_attention_class(attn_type: str, use_extend_attention: bool):
    if attn_type == "gqa":
        if use_extend_attention:
            return PagedExtendAttention
        return PagedGQAttention

    elif attn_type == "mla":
        return PagedMLAttention

    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")
