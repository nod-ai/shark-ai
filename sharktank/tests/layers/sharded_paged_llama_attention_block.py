# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from sharktank.layers import (
    PagedLlamaAttentionBlock,
    PagedAttention,
    RotaryEmbeddingLayer,
)
from sharktank.layers.testing import make_llama_attention_block_theta, make_rand_torch
from sharktank.models.llama.sharding import PagedLlamaAttentionBlockSharding
from sharktank.types import SplitPrimitiveTensor, unbox_tensor
import torch
from sharktank import ops
from copy import deepcopy
import pytest


class ShardedPagedLlamaAttentionBlockTest(unittest.TestCase):
    """Verify that the sharded Llama paged attention block behaves in PyTorch as the
    unsharded variant."""

    def setUp(self):
        torch.manual_seed(12345)
        self.transformer_block_count = 13
        self.block_index = 1
        self.shard_count = 3
        self.head_count_kv = 2 * self.shard_count
        self.attention_head_count = 5 * self.head_count_kv
        self.attention_head_dim = 11 * 2
        self.rms_epsilon = 0.01
        self.block_seq_stride = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.embedding_length = self.attention_head_count * self.attention_head_dim
        self.rope_dimension_count = self.attention_head_dim
        self.block_seqlen = 7
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = None
        self.batch_size = 3
        self.start_index = 0

    def testSmallSizedLayerFp64(self):
        self.runTestSmallSizedLayer(dtype=torch.float64)

    @pytest.mark.xfail(
        reason="The accuracy seems low (atol=0.0018, rtol=0.5065)",
        strict=True,
        raises=AssertionError,
    )
    def testSmallSizedLayerFp32(self):
        self.runTestSmallSizedLayer(dtype=torch.float32)

    def runTestSmallSizedLayer(self, dtype: torch.dtype):
        torch.set_default_dtype(dtype)

        def make_paged_kv_cache(shard_count: int) -> PagedAttention:
            return PagedAttention(
                transformer_block_count=self.transformer_block_count,
                attn_head_count=self.head_count_kv,
                attn_head_dim=self.attention_head_dim,
                cache_partition_count=self.cache_partition_count,
                block_seq_stride=self.block_seq_stride,
                cache_dtype=dtype,
                attn_dtype=dtype,
                shard_count=shard_count,
            )

        cache = make_paged_kv_cache(shard_count=1)
        sharded_cache = make_paged_kv_cache(shard_count=self.shard_count)

        def make_unsharded_and_sharded_equal_cache_states() -> (
            tuple[list[torch.Tensor], list[SplitPrimitiveTensor]]
        ):
            cache_state = cache.allocate(self.page_count)
            cache_state[0] = make_rand_torch(cache_state[0].shape, dtype=dtype)
            sharded_cache_state = sharded_cache.shard_state(deepcopy(cache_state))
            return cache_state, sharded_cache_state

        (
            cache_state,
            sharded_cache_state,
        ) = make_unsharded_and_sharded_equal_cache_states()

        input_tensor = make_rand_torch(
            (
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ),
            dtype=dtype,
        )
        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )
        embedding_module = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
        )

        theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )
        attention_block = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_index,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
        )
        expected_result = attention_block(
            input_tensor,
            embedding=embedding_module,
            seq_block_ids=seq_block_ids,
            start_index=self.start_index,
            cache_state=cache_state,
        )

        sharded_input_tensor = ops.replicate(input_tensor, count=self.shard_count)
        sharded_seq_block_ids = ops.replicate(seq_block_ids, count=self.shard_count)
        sharded_embedding_module = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
            tensor_parallelism_size=self.shard_count,
        )

        theta_sharding = PagedLlamaAttentionBlockSharding(shard_count=self.shard_count)
        sharded_theta = ops.reshard(theta, theta_sharding)
        sharded_attention_block = PagedLlamaAttentionBlock(
            theta=sharded_theta,
            block_index=self.block_index,
            cache=sharded_cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
        )
        sharded_result = sharded_attention_block(
            sharded_input_tensor,
            embedding=sharded_embedding_module,
            seq_block_ids=sharded_seq_block_ids,
            start_index=self.start_index,
            cache_state=sharded_cache_state,
        )

        actual_result = unbox_tensor(ops.unshard(sharded_result))
        actual_cache_state = unbox_tensor(
            ops.unshard(
                sharded_cache.unflatten_page_table(sharded_cache_state)
            ).flatten(start_dim=1)
        )

        torch.testing.assert_close(actual_result, expected_result)
        torch.testing.assert_close(actual_cache_state, cache_state[0])
