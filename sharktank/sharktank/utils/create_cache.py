# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers import *
from sharktank.types.quantizers import StaticScaledQuantizer


def create_paged_attention(
    config: "LlamaModelConfig",
    kv_cache: KVCache,
    use_rope: bool,
    block_index: int,
    k_quantizer: StaticScaledQuantizer | None = None,
    v_quantizer: StaticScaledQuantizer | None = None,
) -> PagedAttention:
    # TODO: Add deepseek PagedLatentAttention

    if config.kv_cache_type != "paged":
        raise ValueError("Model does not use paged kv cache, cannot create kv cache")

    hp = config.hp
    return PagedAttention(
        attention_chunk_size=config.attention_chunk_size,
        transformer_block_index=block_index,
        attn_type=attn_type_map[hp.model_arch],
        kv_cache=kv_cache,
        use_rope=use_rope,
        attn_dtype=config.attention_dtype,
        activation_dtype=config.activation_dtype,
        k_quantizer=k_quantizer,
        v_quantizer=v_quantizer,
    )
