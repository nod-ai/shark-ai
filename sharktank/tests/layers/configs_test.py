# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.layers.configs import LlamaHParams


def test_llama_hp_params_to_from_gguf_props_roundtrip():
    params = LlamaHParams(
        model_arch="llama",
        context_length=1,
        embedding_length=2,
        block_count=3,
        feed_forward_length=3,
        rope_dimension_count=4,
        rope_freq_base=5.0,
        attention_head_count=6,
        attn_head_dim=4,
        attention_layer_norm_rms_epsilon=8.0,
        attention_head_count_kv=9,
        expert_count=10,
        expert_used_count=11,
        n_dense_layers=0,
    )
    roundtripped_params = LlamaHParams.from_gguf_props(params.to_gguf_props())
    assert params == roundtripped_params
