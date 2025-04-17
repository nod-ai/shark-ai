# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.layers.configs.llm_configs import *
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer
from sharktank.layers.paged_attention import PagedAttention
from sharktank.models.llm import AttentionFFNBlock
from sharktank.models.llama.testing import *

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig


class AttentionBlockTest(unittest.TestCase):
    def test(self):
        torch.manual_seed(1234567)
        torch.set_default_dtype(torch.float32)
        block_index = 0
        seq_len = 13
        head_count = 32
        head_dim = 100
        hidden_size = 3200
        ffn_dim = 8640
        head_count_kv = 32
        block_seq_stride = 1
        rms_epsilon = 0.01
        rope_dimension_count = 100
        rope_freq_base = 10000.0
        max_seq_len = 2048
        attention_block_theta = make_attention_block_theta(
            feature_dim=head_count * head_dim, ffn_dim=ffn_dim, dtype=torch.float32
        )

        hp = LlamaHParams(
            model_arch="llama",
            context_length=max_seq_len,
            embedding_length=head_count * head_dim,
            block_count=1,
            feed_forward_length=ffn_dim,
            attention_head_count=head_count,
            attention_layer_norm_rms_epsilon=rms_epsilon,
            attention_head_count_kv=head_count_kv,
            attn_head_dim=head_dim,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=rope_freq_base,
        )

        llama_config = LlamaModelConfig(
            hp,
            attention_kernel="torch",
            block_seq_stride=block_seq_stride,
        )

        paged_kv_cache = PagedAttention(
            transformer_block_count=head_count,
            attn_head_count=head_count,
            attn_head_dim=head_dim,
            cache_partition_count=2,  # One for each of K/V.
            block_seq_stride=block_seq_stride,
            device="cpu",
            dtype=torch.float32,
        )
        attention_block = AttentionFFNBlock(
            theta=attention_block_theta,
            block_index=block_index,
            cache=paged_kv_cache,
            config=llama_config,
        )
        attention_embedding = RotaryEmbeddingLayer(
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=rope_freq_base,
            max_seqlen=max_seq_len,
            device="cpu",
            use_hf=True,
        )
        position_embeddings = attention_embedding.rotary_embed_table
        input_tensor = make_rand_torch(
            (1, seq_len, head_count * head_dim), dtype=torch.float32
        )

        sharktank_output = attention_block(
            input_tensor,
            embedding=attention_embedding,
            attention_mask=torch.zeros(1, seq_len, seq_len, dtype=torch.float32),
            start_index=0,
            cache_state=paged_kv_cache.allocate(128),
            seq_block_ids=torch.arange(seq_len).view(1, -1),
        )

        llama_config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=head_count,
            num_key_value_heads=head_count_kv,
            max_position_embeddings=max_seq_len,
            rms_norm_eps=rms_epsilon,
            rope_theta=10000,
        )
        llama_attention_block = LlamaAttention(
            config=llama_config, layer_idx=block_index
        )

        llama_attention_block.q_proj.weight = torch.nn.Parameter(
            attention_block_theta("attn_q.weight").as_torch(),
            requires_grad=True,
        )
        llama_attention_block.k_proj.weight = torch.nn.Parameter(
            attention_block_theta("attn_k.weight").as_torch(),
            requires_grad=True,
        )
        llama_attention_block.v_proj.weight = torch.nn.Parameter(
            attention_block_theta("attn_v.weight").as_torch(),
            requires_grad=True,
        )
        llama_attention_block.o_proj.weight = torch.nn.Parameter(
            attention_block_theta("attn_output.weight").as_torch(),
            requires_grad=True,
        )

        llama_mlp = LlamaMLP(config=llama_config)
        llama_mlp.gate_proj.weight = torch.nn.Parameter(
            attention_block_theta("ffn_gate.weight").as_torch(), requires_grad=True
        )
        llama_mlp.up_proj.weight = torch.nn.Parameter(
            attention_block_theta("ffn_up.weight").as_torch(), requires_grad=True
        )
        llama_mlp.down_proj.weight = torch.nn.Parameter(
            attention_block_theta("ffn_down.weight").as_torch(), requires_grad=True
        )

        llama_input_layernorm = LlamaRMSNorm(hidden_size=hidden_size, eps=rms_epsilon)
        llama_input_layernorm.weight = torch.nn.Parameter(
            attention_block_theta("attn_norm.weight").as_torch(),
            requires_grad=True,
        )

        llama_post_attention_layernorm = LlamaRMSNorm(
            hidden_size=hidden_size, eps=rms_epsilon
        )
        llama_post_attention_layernorm.weight = torch.nn.Parameter(
            attention_block_theta("ffn_norm.weight").as_torch(),
            requires_grad=True,
        )

        llama_decoder_layer = LlamaDecoderLayer(
            config=llama_config, layer_idx=block_index
        )
        llama_decoder_layer.self_attn = llama_attention_block
        llama_decoder_layer.mlp = llama_mlp
        llama_decoder_layer.input_layernorm = llama_input_layernorm
        llama_decoder_layer.post_attention_layernorm = llama_post_attention_layernorm
        position_embeddings = [x[:seq_len, :].unsqueeze(0) for x in position_embeddings]
        huggingface_output = llama_decoder_layer(
            input_tensor,
            position_embeddings=position_embeddings,
        )[0]
        assert sharktank_output.shape == huggingface_output.shape
        torch.testing.assert_close(
            sharktank_output, huggingface_output, atol=1e-5, rtol=5e-1
        )


if __name__ == "__main__":
    unittest.main()
