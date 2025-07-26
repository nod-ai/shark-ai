# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from copy import deepcopy

import torch

from sharktank.layers import build_rotary_layer
from sharktank.layers.paged_llama_attention_block import PagedLlamaAttentionBlock
from sharktank.layers.testing import make_rand_torch

from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.types.theta import Theta, flat_to_nested_dict
from sharktank.types import unbox_tensor, SplitPrimitiveTensor
from sharktank.types.sharding import shard_theta, LatentAttentionBlockSharding
from sharktank import ops
from sharktank.utils.create_cache import *
from sharktank.utils import iterables_equal
from sharktank.utils.testing import assert_tensor_close


class ShardedPagedLatentAttentionBlockTest(unittest.TestCase):
    """Verify that the sharded latent paged attention block behaves in PyTorch as the
    unsharded variant."""

    def testShardedLatentLayer(self):
        rtol = 1e-5
        atol = 1e-5

        bs = 1
        start_index = 0
        block_seqlen = 7
        tensor_parallelism_size = 2
        page_count = 64

        # This needs to be f32 because we are generating a config for KV cache of f32.
        dtype = torch.float32

        theta, config = generate(12345)
        theta = theta("blk", 0)
        hp = config.hp
        config.block_seq_stride = 16
        embedding_length = hp.embedding_length
        max_seqlen = config.block_seq_stride * block_seqlen

        spec = LatentAttentionBlockSharding(shard_count=tensor_parallelism_size)
        sharding_keys = {k for k in spec.theta_sharding().keys()}
        flattened = theta.flatten()
        t = {}
        for k in flattened:
            if k.split(".")[0] in sharding_keys:
                t[k] = flattened[k]

        theta = Theta(flat_to_nested_dict(t))

        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = tensor_parallelism_size
        sharded_theta = shard_theta(theta=theta, sharding=spec)

        input_tensor = torch.rand((bs, max_seqlen, embedding_length))
        seq_block_ids = torch.arange(bs * block_seqlen).view(bs, -1)
        sharded_input_tensor = ops.replicate(
            input_tensor, count=tensor_parallelism_size
        )
        sharded_seq_block_ids = ops.replicate(
            seq_block_ids, count=tensor_parallelism_size
        )

        cache = create_paged_kv_cache(config)
        sharded_cache = create_paged_kv_cache(sharded_config)

        def assert_equal_unsharded_and_sharded_cache_states(
            cache_state: list[torch.Tensor],
            sharded_cache_state: list[SplitPrimitiveTensor],
        ):
            cache_state = cache.unshard_state(cache_state)[0]
            sharded_state_as_unsharded = sharded_cache.unshard_state(
                sharded_cache_state
            )[0]
            assert iterables_equal(sharded_state_as_unsharded.shape, cache_state.shape)
            assert ops.equal(
                cache_state,
                sharded_state_as_unsharded,
            )

        def assert_close_unsharded_and_sharded_cache_states(
            cache_state: list[torch.Tensor],
            sharded_cache_state: list[SplitPrimitiveTensor],
        ):
            cache_state = cache.unshard_state(cache_state)[0]
            sharded_state_as_unsharded = sharded_cache.unshard_state(
                sharded_cache_state
            )[0]
            assert iterables_equal(sharded_state_as_unsharded.shape, cache_state.shape)
            assert_tensor_close(
                cache_state,
                sharded_state_as_unsharded,
                rtol=rtol,
                atol=atol,
            )

        def make_unsharded_and_sharded_equal_cache_states() -> (
            tuple[list[torch.Tensor], list[SplitPrimitiveTensor]]
        ):
            cache_state = cache.allocate(page_count)
            cache_state[0] = make_rand_torch(cache_state[0].shape, dtype=dtype)
            sharded_cache_state = sharded_cache.shard_state(deepcopy(cache_state))
            assert_equal_unsharded_and_sharded_cache_states(
                cache_state, sharded_cache_state
            )
            return cache_state, sharded_cache_state

        (
            cache_state,
            sharded_cache_state,
        ) = make_unsharded_and_sharded_equal_cache_states()

        embedding = build_rotary_layer(
            rope_dimension_count=hp.rope_dimension_count,
            rope_freq_base=hp.rope_freq_base,
        )

        reference_model = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=0,
            cache=cache,
            head_count=hp.attention_head_count,
            head_dim=hp.attn_head_dim,
            head_count_kv=hp.attention_head_count_kv,
            rms_epsilon=hp.attention_layer_norm_rms_epsilon,
            rope_dimension_count=hp.rope_dimension_count,
            v_head_dim=hp.v_head_dim,
            model_arch=hp.model_arch,
        )

        expected_result = reference_model(
            input_tensor,
            embedding=embedding,
            seq_block_ids=seq_block_ids,
            start_index=start_index,
            cache_state=cache_state,
        )

        sharded_embedding = build_rotary_layer(
            rope_dimension_count=hp.rope_dimension_count,
            rope_freq_base=hp.rope_freq_base,
            tensor_parallelism_size=tensor_parallelism_size,
        )

        sharded_model = PagedLlamaAttentionBlock(
            theta=sharded_theta,
            block_index=0,
            cache=sharded_cache,
            head_count=hp.attention_head_count,
            head_dim=hp.attn_head_dim,
            head_count_kv=hp.attention_head_count_kv,
            rms_epsilon=hp.attention_layer_norm_rms_epsilon,
            rope_dimension_count=hp.rope_dimension_count,
            v_head_dim=hp.v_head_dim,
            model_arch=hp.model_arch,
        )

        sharded_result = sharded_model(
            sharded_input_tensor,
            embedding=sharded_embedding,
            seq_block_ids=sharded_seq_block_ids,
            start_index=start_index,
            cache_state=sharded_cache_state,
        )

        actual_result = unbox_tensor(ops.unshard(sharded_result))

        assert_tensor_close(actual_result, expected_result, rtol=rtol, atol=atol)
        assert_close_unsharded_and_sharded_cache_states(
            cache_state, sharded_cache_state
        )
