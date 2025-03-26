# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from iree.turbine import aot
from sharktank.layers import (
    PagedLlamaAttentionBlock,
    PagedAttention,
    RotaryEmbeddingLayer,
)
from sharktank.layers.testing import make_llama_attention_block_theta
from sharktank.types.tensors import DefaultPrimitiveTensor


class PagedLlamaAttentionBlockTest(unittest.TestCase):
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

    def testExportDecomposed(self):
        dtype = torch.float32

        cache = PagedAttention(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.head_count_kv,
            attn_head_dim=self.attention_head_dim,
            cache_partition_count=self.cache_partition_count,
            block_seq_stride=self.block_seq_stride,
            dtype=dtype,
        )

        cache_state = cache.allocate(self.page_count)
        cache_state[0] = torch.rand(cache_state[0].shape, dtype=dtype)

        theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )
        attn = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_index,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
            attention_kernel="decomposed",
        )

        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )

        embedding_module = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
        )

        class MyModule(torch.nn.Module):
            def forward(self, h, seq_block_ids, cache_state):
                return attn.forward(
                    h,
                    seq_block_ids=seq_block_ids,
                    embedding=embedding_module,
                    start_index=0,
                    cache_state=cache_state,
                )

        mod = MyModule()
        h = torch.rand(
            [
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ]
        )
        mod.forward(h, seq_block_ids, cache_state)
        ep = torch.export.export(
            mod,
            args=(
                h,
                seq_block_ids,
                cache_state,
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertNotIn("scaled_dot_product_attention", asm)

    @pytest.mark.xfail(
        torch.__version__ >= (2, 4),
        reason="https://github.com/nod-ai/shark-ai/issues/684",
    )
    @pytest.mark.skipif(
        torch.__version__ >= (2, 5),
        reason="https://github.com/nod-ai/shark-ai/issues/684, error slows down CI",
    )
    def testExportNondecomposed(self):
        dtype = torch.float32

        cache = PagedAttention(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.head_count_kv,
            attn_head_dim=self.attention_head_dim,
            cache_partition_count=self.cache_partition_count,
            block_seq_stride=self.block_seq_stride,
            dtype=dtype,
        )

        cache_state = cache.allocate(self.page_count)
        cache_state[0] = torch.rand(cache_state[0].shape, dtype=dtype)

        theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )
        attn = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_index,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
            attention_kernel="torch",
        )

        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )

        embedding_module = RotaryEmbeddingLayer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
        )

        class MyModule(torch.nn.Module):
            def forward(self, h, seq_block_ids, cache_state):
                return attn.forward(
                    h,
                    seq_block_ids=seq_block_ids,
                    embedding=embedding_module,
                    start_index=0,
                    cache_state=cache_state,
                )

        mod = MyModule()
        h = torch.rand(
            [
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ]
        )
        mod.forward(h, seq_block_ids, cache_state)
        ep = torch.export.export(
            mod,
            args=(
                h,
                seq_block_ids,
                cache_state,
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("torch.aten._scaled_dot_product_flash_attention_for_cpu", asm)


if __name__ == "__main__":
    unittest.main()
