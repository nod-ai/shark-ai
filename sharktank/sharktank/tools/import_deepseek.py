# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import argparse
import dataclasses
import json
import logging
from typing import Literal
from tqdm import tqdm
import torch

from safetensors.torch import safe_open
from sharktank.layers.configs.llm_configs import LlamaHParams
from sharktank.types import Dataset, Theta
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.utils import cli
from sharktank.utils.misc import get_files

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    context_length: int = 163840
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attn_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


baseMapping = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

attnMapping = {
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_b_proj.weight": "attn_kv_b.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "ffn_gate_e_score_correction_bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
}

expertMapping = {
    "gate_proj.weight": "ffn_gate_exps.weight",
    "down_proj.weight": "ffn_down_exps.weight",
    "up_proj.weight": "ffn_up_exps.weight",
}

# Multi-Token Prediction (MTP) Modules
mtpMapping = {
    "embed_tokens.weight": "mtp_embed_tokens.weight",
    "enorm.weight": "mtp_enorm.weight",
    "hnorm.weight": "mtp_hnorm.weight",
    "eh_proj.weight": "mtp_eh_proj.weight",
    "shared_head.norm.weight": "mtp_shared_head.norm.weight",
    "shared_head.head.weight": "mtp_shared_head.head.weight",
}

if __name__ == "__main__":

    """
    Import Deepseek model from safetensors to irpa
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--safetensors", type=str, required=True)
    parser.add_argument("--irpa-path", type=str, required=True)
    parser.add_argument("--json-path", type=str)

    cli.add_log_options(parser)

    args = parser.parse_args()
    logger.setLevel(args.loglevel)

    torch.set_num_threads(16)

    config = json.load(open(args.config, "r"))
    modelargs = ModelArgs(**config)
    hp = LlamaHParams(
        model_arch="deepseek2",
        context_length=modelargs.context_length,
        embedding_length=modelargs.dim,
        vocab_size=modelargs.vocab_size,
        block_count=modelargs.n_layers,
        feed_forward_length=modelargs.inter_dim,
        attention_head_count=modelargs.n_heads,
        attn_head_dim=modelargs.attn_head_dim,
        attention_layer_norm_rms_epsilon=1e-6,
        attention_head_count_kv=128,
        q_lora_rank=modelargs.q_lora_rank,
        kv_lora_rank=modelargs.kv_lora_rank,
        qk_nope_head_dim=modelargs.qk_nope_head_dim,
        qk_rope_head_dim=modelargs.qk_rope_head_dim,
        v_head_dim=modelargs.v_head_dim,
        rope_dimension_count=modelargs.qk_rope_head_dim,
        rope_freq_base=modelargs.rope_theta,
        expert_feed_forward_length=modelargs.moe_inter_dim,
        expert_count=modelargs.n_routed_experts,
        expert_used_count=modelargs.n_activated_experts,
        expert_shared_count=modelargs.n_shared_experts,
        n_expert_groups=modelargs.n_expert_groups,
        n_limited_groups=modelargs.n_limited_groups,
        n_dense_layers=modelargs.n_dense_layers,
        route_scale=modelargs.route_scale,
        yarn_beta_slow=modelargs.beta_slow,
        yarn_beta_fast=modelargs.beta_fast,
        yarn_factor=modelargs.rope_factor,
        yarn_original_context_len=modelargs.original_seq_len,
        yarn_mscale=modelargs.mscale,
    )

    st_path = args.safetensors
    irpa_path = args.irpa_path
    tensors = {}
    layers = {}

    safetensors_list = get_files(st_path, ".safetensors")
    logger.info(f"# safetensors files: {len(safetensors_list)}")

    for file_path in tqdm(
        safetensors_list,
        desc=f"Loading safetensors: ",
    ):
        with safe_open(file_path, framework="pt", device="cpu") as st:

            for key in baseMapping:
                try:
                    tensors[baseMapping[key]] = st.get_tensor(key)
                except:
                    continue

            for key in st.keys():
                parts = key.split(".", 3)
                if parts[1] != "layers" or "weight_scale_inv" in parts[-1]:
                    continue
                layer = int(parts[2])
                layer_name = parts[-1]
                if layer not in layers:
                    layers[layer] = {}
                layers[layer][layer_name] = st.get_tensor(key)

    logger.info(f"Base layers mapping completed: {len(tensors.keys())}")
    logger.info(f"Model layers mapping completed: {len(layers.keys())}")

    for layer in tqdm(
        layers,
        mininterval=300,
        desc=f"Saving to irpa: ",
    ):
        weights = layers[layer]
        experts = {}
        for name in weights:
            logger.debug(f"layer, name: {layer, name}")
            weight = weights[name]
            if name in baseMapping:
                continue
            elif name in attnMapping:
                tensors[f"blk.{layer}.{attnMapping[name]}"] = weight
                continue
            elif name in mtpMapping:
                tensors[f"blk.{layer}.{mtpMapping[name]}"] = weight
                continue
            elif name.startswith("mlp.experts."):
                split = name.split(".", 3)
                id = int(split[2])
                if id not in experts:
                    experts[id] = {}
                experts[id][split[-1]] = weight
                continue
            else:
                assert False and "unhandled tensor found"

        expert_keys = experts[0].keys() if experts else []
        for key in expert_keys:
            exs = [experts[expert][key] for expert in experts]
            weight = torch.stack(exs, dim=0)
            for t in exs:
                del t
            tensors[f"blk.{layer}.{expertMapping[key]}"] = weight

    config_dict = hp.to_gguf_props()

    tensors = [
        DefaultPrimitiveTensor(name=name, data=tensors[name]) for name in tensors.keys()
    ]
    theta = Theta(tensors)

    dataset = Dataset(config_dict, theta)

    for tensor in dataset.root_theta.flatten().values():
        logger.debug(f"  {tensor.name}: {tensor.shape}, {tensor.dtype}")

    logger.debug(f"config_dict:  {config_dict}")
    dataset.save(irpa_path, io_report_callback=logger.info)
