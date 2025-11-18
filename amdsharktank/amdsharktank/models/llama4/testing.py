# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...layers import BaseCausalLMModel, LlamaModelConfig, LlamaHParams
from ...types import Theta, unbox_tensor
from transformers.models.llama4 import Llama4TextConfig

import torch
from amdsharktank import ops


def config_to_hugging_face_text_config(config: LlamaModelConfig) -> Llama4TextConfig:
    moe_layers = None
    if config.hp.interleave_moe_layer_step is None:
        moe_layers = config.moe_layers
    rope_scaling = {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }
    # Hugging Face hardcodes RoPE layers.
    assert list(config.rope_layers) == [
        i for i in range(config.hp.block_count) if int((i + 1) % 4 != 0)
    ]
    return Llama4TextConfig(
        vocab_size=config.hp.vocab_size,
        hidden_size=config.hp.embedding_length,
        intermediate_size=config.hp.expert_feed_forward_length,
        intermediate_size_mlp=config.hp.feed_forward_length,
        num_hidden_layers=config.hp.block_count,
        num_attention_heads=config.hp.attention_head_count,
        num_key_value_heads=config.hp.attention_head_count_kv,
        head_dim=config.hp.attn_head_dim,
        hidden_act="silu",
        max_position_embeddings=config.hp.context_length,
        rms_norm_eps=config.hp.attention_layer_norm_rms_epsilon,
        rope_theta=config.hp.rope_freq_base,
        num_experts_per_tok=config.hp.expert_used_count,
        num_local_experts=config.hp.expert_count,
        moe_layers=moe_layers,
        interleave_moe_layer_step=config.hp.interleave_moe_layer_step,
        rope_scaling=rope_scaling,
        attention_chunk_size=config.attention_chunk_size,
        torch_dtype=config.dtype,
        attn_temperature_tuning=config.hp.attn_temperature_tuning,
        floor_scale=config.hp.floor_scale,
        attention_scale=config.hp.attention_scale,
        attn_implementation="sdpa",
    )


def block_theta_to_hugging_face_state_dict(
    theta: Theta, config: LlamaModelConfig, block_idx: int
) -> dict[str, torch.Tensor]:
    name_map = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        # MoE
        "ffn_gate_inp.weight": "feed_forward.router.weight",
        "shared_experts.ffn_gate.weight": "feed_forward.shared_expert.gate_proj.weight",
        "shared_experts.ffn_up.weight": "feed_forward.shared_expert.up_proj.weight",
        "shared_experts.ffn_down.weight": "feed_forward.shared_expert.down_proj.weight",
    }
    if block_idx in config.moe_layers:
        name_map.update(
            {
                "ffn_gate_shexp.weight": "feed_forward.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight": "feed_forward.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight": "feed_forward.shared_expert.down_proj.weight",
            }
        )
    else:
        name_map.update(
            {
                "ffn_gate.weight": "feed_forward.gate_proj.weight",
                "ffn_up.weight": "feed_forward.up_proj.weight",
                "ffn_down.weight": "feed_forward.down_proj.weight",
            }
        )

    res = {name_map[k]: v for k, v in theta.flatten().items() if k in name_map}
    if "ffn_gate_exps" in theta.tree:
        res["feed_forward.experts.gate_up_proj"] = ops.cat(
            [
                unbox_tensor(theta("ffn_gate_exps.weight")),
                unbox_tensor(theta("ffn_up_exps.weight")),
            ],
            dim=1,
        ).transpose(1, 2)
        res["feed_forward.experts.down_proj"] = unbox_tensor(
            theta("ffn_down_exps.weight")
        ).transpose(1, 2)
    return res


def theta_to_hugging_face_state_dict(
    theta: Theta, config: LlamaModelConfig
) -> dict[str, torch.Tensor]:
    res = {
        "model.embed_tokens.weight": theta("token_embd.weight"),
        "model.norm.weight": theta("output_norm.weight").squeeze(0),
        "lm_head.weight": theta("output.weight"),
    }
    i = 0
    blocks_theta = theta("blk")
    while f"{i}" in blocks_theta.tree:
        block_state_dict = block_theta_to_hugging_face_state_dict(
            blocks_theta(i), config=config, block_idx=i
        )
        res.update({f"model.layers.{i}.{k}": v for k, v in block_state_dict.items()})
        i += 1
    res = {k: unbox_tensor(v) for k, v in res.items()}
    return res


def make_toy_model_config(dtype: torch.dtype) -> LlamaModelConfig:
    attention_head_count_kv = 4
    attention_head_count = attention_head_count_kv * 5
    vocabulary_size = 19

    # When comparing with Hugging Face, its Flex attention requires a power 2.
    rope_dimension_count = 4 * 2

    attn_head_dim = rope_dimension_count
    block_seq_stride = 13
    expert_feed_forward_length = 29
    return LlamaModelConfig(
        hp=LlamaHParams(
            context_length=block_seq_stride * 11,
            embedding_length=attention_head_count * attn_head_dim,
            vocab_size=vocabulary_size,
            block_count=4,
            feed_forward_length=23,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=500000.0,
            rope_interleave_emb=False,
            yarn_beta_slow=1,
            yarn_beta_fast=4,
            yarn_factor=8,
            yarn_original_context_len=8192,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=0.01,
            attention_head_count_kv=attention_head_count_kv,
            expert_feed_forward_length=expert_feed_forward_length,
            expert_count=3,
            expert_used_count=2,
            expert_shared_count=1,
            expert_shared_feed_forward_length=expert_feed_forward_length,
            interleave_moe_layer_step=2,
            model_arch="llama4",
            no_rope_layer_step=4,
            attn_temperature_tuning=True,
            floor_scale=31,
            attention_scale=0.2,
        ),
        block_seq_stride=block_seq_stride,
        activation_dtype=dtype,
        attention_dtype=dtype,
        dtype=dtype,
        use_qk_norm=True,
        attention_chunk_size=37,
    )
