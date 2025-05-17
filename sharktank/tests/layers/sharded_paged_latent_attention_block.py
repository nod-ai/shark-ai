# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from copy import deepcopy

import torch

from sharktank.layers.paged_llama_attention_block import PagedLlamaAttentionBlock
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from sharktank.layers.testing import make_rand_torch, make_latent_attention_block_theta

from sharktank.types import unbox_tensor
from sharktank.types.sharding import LatentAttentionBlockSharding
from sharktank import ops
from sharktank.utils.create_cache import *


class ShardedPagedLatentAttentionBlockTest(unittest.TestCase):
    """Verify that the sharded latent paged attention block behaves in PyTorch as the
    unsharded variant."""

    def setUp(self):
        torch.manual_seed(12345)
        self.dtype = torch.float32
        self.max_blocks = 8
        self.vocabulary_size = 256
        self.model_arch = "deepseek2"
        self.batch_size = 1
        self.start_index = 0

        self.transformer_block_count = 1
        self.block_idx = 0
        self.shard_count = 2
        self.attention_head_count_kv = 4
        self.attention_head_count = 4
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.attention_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.rms_epsilon = 9.0
        self.block_seq_stride = 16
        self.cache_partition_count = 2
        self.page_count = 64
        self.embedding_length = 768
        self.rope_dimension_count = 64
        self.block_seqlen = 7
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = 10000.0

    def testShardedLatentLayer(self):
        def make_paged_kv_cache(shard_count: int) -> PagedAttention:
            return PagedAttention(
                transformer_block_count=self.transformer_block_count,
                attn_head_count=self.attention_head_count_kv,
                attn_head_dim=self.attention_head_dim,
                cache_partition_count=self.cache_partition_count,
                block_seq_stride=self.block_seq_stride,
                cache_dtype=self.dtype,
                attn_dtype=self.dtype,
                shard_count=shard_count,
            )

        input_tensor = make_rand_torch(
            (
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ),
            dtype=self.dtype,
        )
        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )
        sharded_input_tensor = ops.replicate(input_tensor, count=self.shard_count)
        sharded_seq_block_ids = ops.replicate(seq_block_ids, count=self.shard_count)

        cache = make_paged_kv_cache(shard_count=1)
        sharded_cache = make_paged_kv_cache(shard_count=self.shard_count)
        cache_state = cache.allocate(self.page_count)
        cache_state[0] = make_rand_torch(cache_state[0].shape, dtype=self.dtype)
        sharded_cache_state = sharded_cache.shard_state(deepcopy(cache_state))

        theta = make_latent_attention_block_theta(
            block_idx=self.block_idx,
            head_count=self.attention_head_count,
            head_count_kv=self.attention_head_count_kv,
            embedding_length=self.embedding_length,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_latent_dim=self.kv_lora_rank,
            v_head_dim=self.v_head_dim,
            dtype=self.dtype,
        )

        theta_sharding = LatentAttentionBlockSharding(shard_count=self.shard_count)
        sharded_theta = ops.reshard(theta, theta_sharding)

        embedding = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            rope_freq_base=self.rope_freq_base,
            max_seqlen=self.max_seqlen,
        )

        attention_block = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_idx,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.attention_head_count_kv,
            rms_epsilon=self.rms_epsilon,
            rope_dimension_count=self.rope_dimension_count,
            v_head_dim=self.v_head_dim,
            model_arch=self.model_arch,
        )

        expected_result = attention_block(
            input_tensor,
            embedding=embedding,
            seq_block_ids=seq_block_ids,
            start_index=self.start_index,
            cache_state=cache_state,
        )

        sharded_embedding = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            rope_freq_base=self.rope_freq_base,
            max_seqlen=self.max_seqlen,
            tensor_parallelism_size=self.shard_count,
        )

        sharded_attention_block = PagedLlamaAttentionBlock(
            theta=sharded_theta,
            block_index=self.block_idx,
            cache=sharded_cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.attention_head_count_kv,
            rms_epsilon=self.rms_epsilon,
            rope_dimension_count=self.rope_dimension_count,
            v_head_dim=self.v_head_dim,
            model_arch=self.model_arch,
        )

        sharded_result = sharded_attention_block(
            sharded_input_tensor,
            embedding=sharded_embedding,
            seq_block_ids=sharded_seq_block_ids,
            start_index=self.start_index,
            cache_state=sharded_cache_state,
        )

        actual_result = unbox_tensor(ops.unshard(sharded_result))

        actual_cache_state = unbox_tensor(sharded_cache_state[0])

        torch.testing.assert_close(actual_result, expected_result)
        torch.testing.assert_close(actual_cache_state, cache_state[0])
