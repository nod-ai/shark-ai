import argparse
import dataclasses
import json
import torch

from safetensors.torch import save_file, safe_open
from amdsharktank.layers.configs.llm_configs import LlamaHParams
from amdsharktank.types import Dataset, Theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor
from amdsharktank.utils.logging import get_logger
from typing import Literal


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
    q_lora_rank: int = 0
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
    rope_scaling_yarn_log_multiplier: float = 0.10000000149011612


if __name__ == "__main__":
    logger = get_logger("amdsharktank.tools.import_deepseek")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--safetensors", type=str, required=True)
    parser.add_argument("--irpa-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    args = parser.parse_args()

    config = json.load(open(args.config, "r"))
    modelargs = ModelArgs(**config)
    hp = LlamaHParams(
        model_arch="deepseek2",
        context_length=modelargs.original_seq_len,
        embedding_length=modelargs.vocab_size,
        block_count=modelargs.n_layers,
        feed_forward_length=modelargs.inter_dim,
        attention_head_count=modelargs.n_heads,
        attn_head_dim=modelargs.attn_head_dim,
        attention_layer_norm_rms_epsilon=1e-6,
        attention_head_count_kv=-1,
        q_lora_rank=modelargs.q_lora_rank,
        kv_lora_rank=modelargs.kv_lora_rank,
        qk_nope_head_dim=modelargs.qk_nope_head_dim,
        qk_rope_head_dim=modelargs.qk_rope_head_dim,
        v_head_dim=modelargs.v_head_dim,
        rope_dimension_count=modelargs.qk_rope_head_dim,
        rope_freq_base=modelargs.rope_theta,
        expert_count=modelargs.n_routed_experts,
        expert_used_count=modelargs.n_activated_experts,
        expert_shared_count=modelargs.n_shared_experts,
        n_expert_groups=modelargs.n_expert_groups,
        n_limited_groups=modelargs.n_limited_groups,
        n_dense_layers=modelargs.n_dense_layers,
        route_scale=modelargs.route_scale,
    )

    x = torch.randint(0, modelargs.vocab_size, (2, 16))

    st_path = args.safetensors
    json_path = args.json_path
    irpa_path = args.irpa_path

    st = safe_open(st_path, framework="pt")

    baseMapping = {
        "token_embd.weight": "embed.weight",
        "output_norm.weight": "norm.weight",
        "output.weight": "head.weight",
    }

    attnMapping = {
        "attn.kv_norm.weight": "kv_norm.weight",
        "attn.wkv_a.weight": "wkv_a.weight",
        "attn.wkv_b.weight": "wkv_b.weight",
        "attn.wo.weight": "wo.weight",
        "attn.wq.weight": "wq.weight",
        "attn.wq_a.weight": "wq_a.weight",
        "attn.wq_b.weight": "wq_b.weight",
        "attn.q_norm.weight": "q_norm.weight",
        "attn_norm.weight": "attn_norm.weight",
        "ffn.w1.weight": "ffn.ffn_gate.weight",
        "ffn.w2.weight": "ffn.ffn_down.weight",
        "ffn.w3.weight": "ffn.ffn_up.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "ffn.gate.weight": "moe.ffn_gate_inp.weight",
        "ffn.shared_experts.w1.weight": "moe.shared_experts.ffn_gate.weight",
        "ffn.shared_experts.w2.weight": "moe.shared_experts.ffn_down.weight",
        "ffn.shared_experts.w3.weight": "moe.shared_experts.ffn_up.weight",
    }

    expertMapping = {
        "w1.weight": "ffn_gate_exps.weight",
        "w2.weight": "ffn_down_exps.weight",
        "w3.weight": "ffn_up_exps.weight",
    }

    tensors = {}
    for key in baseMapping:
        tensors[key] = st.get_tensor(baseMapping[key])

    layers = {}
    for key in st.keys():
        parts = key.split(".", 2)
        if parts[0] != "layers":
            continue
        layer = int(parts[1])
        if layer not in layers:
            layers[layer] = {}
        layers[layer][parts[2]] = st.get_tensor(key)

    for layer in layers:
        weights = layers[layer]
        experts = {}
        for name in weights:
            weight = weights[name]
            if name in attnMapping:
                tensors[f"blk.{layer}.{attnMapping[name]}"] = weight
                continue

            if name.startswith("ffn.experts."):
                split = name.split(".", 3)
                id = int(split[2])
                if id not in experts:
                    experts[id] = {}
                experts[id][split[3]] = weight
                continue
            assert False and "unhandled tensor found"

        expert_keys = experts[0].keys() if experts else []
        for key in expert_keys:
            exs = [experts[expert][key] for expert in experts]
            tensor = torch.stack(exs, dim=0)
            for t in exs:
                del t
            newKey = expertMapping[key]
            tensors[f"blk.{layer}.moe.{newKey}"] = tensor

    config_json = dataclasses.asdict(hp)
    meta_params = {k: v for k, v in config_json.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json.items() if not k.startswith("_")}
    props = {
        "meta": meta_params,
        "hparams": hparams,
    }

    tensors = [
        DefaultPrimitiveTensor(name=name, data=tensors[name]) for name in tensors.keys()
    ]
    theta = Theta(tensors)

    dataset = Dataset(props, theta)
    dataset.save(irpa_path, io_report_callback=logger.info)
