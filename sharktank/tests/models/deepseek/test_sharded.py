# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers.paged_llama_attention_block import PagedLlamaAttentionBlock
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from sharktank.models.deepseek.sharding import (
    flat_to_nested_dict,
    shard_theta,
    LatentAttentionBlockSharding,
)
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank import ops
from sharktank.types.theta import Theta

import pytest
import torch


@pytest.mark.xfail(
    reason="Deepseek support will be added soon",
)
def test_deepseek():
    theta, config = generate(12345)
    theta = theta("blk", 0)

    sharding = 2
    spec = LatentAttentionBlockSharding(sharding)

    keys = {k for k in theta.keys}
    sharding_keys = {k for k in spec.theta_sharding().keys()}
    flattened = theta.flatten()

    t = {}
    for k in flattened:
        if k.split(".")[0] in sharding_keys:
            t[k] = flattened[k]
    theta = Theta(flat_to_nested_dict(t))

    sharded_theta = shard_theta(theta, spec)

    hp = config.hp
    reference_model = PagedLlamaAttentionBlock(
        theta=theta,
        block_index=0,
        cache=None,
        head_count=hp.attention_head_count,
        head_dim=hp.attn_head_dim,
        head_count_kv=hp.attention_head_count_kv,
        rms_epsilon=hp.attention_layer_norm_rms_epsilon,
        rope_dimension_count=hp.rope_dimension_count,
    )

    sharded_model = PagedLlamaAttentionBlock(
        theta=sharded_theta,
        block_index=0,
        cache=None,
        head_count=hp.attention_head_count,
        head_dim=hp.attn_head_dim,
        head_count_kv=hp.attention_head_count_kv,
        rms_epsilon=hp.attention_layer_norm_rms_epsilon,
        rope_dimension_count=hp.rope_dimension_count,
    )

    bs = 1
    seq = 11
    embed = hp.embedding_length
    input = torch.rand((bs, seq, embed))

    embedding = RotaryEmbeddingLayer(
        rope_dimension_count=hp.rope_dimension_count,
        rope_freq_base=hp.rope_freq_base,
        max_seqlen=hp.context_length,
    )

    sharded_embedding = RotaryEmbeddingLayer(
        rope_dimension_count=hp.rope_dimension_count,
        rope_freq_base=hp.rope_freq_base,
        max_seqlen=hp.context_length,
        tensor_parallelism_size=sharding,
    )

    reference = reference_model.forward(embedding=embedding, h=input)
    sharded = sharded_model.forward(
        embedding=sharded_embedding, h=ops.replicate(input, count=sharding)
    )
    sharded = ops.unshard(sharded)
    assert torch.isclose(reference, sharded, atol=1e-5).all()
