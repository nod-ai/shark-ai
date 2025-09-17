import argparse
import dataclasses
import json
import logging
import torch

from safetensors.torch import save_file, safe_open
from sharktank.layers.configs.llm_configs import LlamaHParams
from sharktank.types import Dataset, Theta
from sharktank.types.tensors import DefaultPrimitiveTensor
from typing import Literal


@dataclasses.dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:

    """

    max_seq_len: int = 4096 * 4
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: int = 150000
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: int = 1
    rope_ntk_beta: int = 32


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--safetensors", type=str, required=True)
    parser.add_argument("--irpa-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--quantize_moe", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config, "r"))
    modelargs = ModelArgs(**config)
    hp = LlamaHParams(
        model_arch="gpt-oss",
        block_count=modelargs.num_hidden_layers,
        expert_count=modelargs.num_experts,
        expert_used_count=modelargs.experts_per_token,
        vocab_size=modelargs.vocab_size,
        embedding_length=modelargs.hidden_size,
        feed_forward_length=modelargs.intermediate_size,
        attn_head_dim=modelargs.head_dim,
        attention_head_count=modelargs.num_attention_heads,
        attention_head_count_kv=modelargs.num_key_value_heads,
        sliding_window=modelargs.sliding_window,
        context_length=modelargs.initial_context_length,
        yarn_original_context_len=modelargs.initial_context_length,
        rope_freq_base=modelargs.rope_theta,
        yarn_factor=modelargs.rope_scaling_factor,
        yarn_beta_slow=modelargs.rope_ntk_alpha,
        yarn_beta_fast=modelargs.rope_ntk_beta,
        attention_layer_norm_rms_epsilon=1e-5,
        rope_dimension_count=modelargs.head_dim,
        swiglu_limit=modelargs.swiglu_limit,
    )

    x = torch.randint(0, modelargs.vocab_size, (2, 16))

    st_path = args.safetensors
    json_path = args.json_path
    irpa_path = args.irpa_path

    st = safe_open(st_path, framework="pt")
    baseMapping = {
        "token_embd.weight": "embedding.weight",
        "output_norm.weight": "norm.scale",
        "output.weight": "unembedding.weight",
    }

    attnMapping = {
        # base attention
        "attn.norm.scale": "attn_norm.weight",
        "attn.out.weight": "attn_output.weight",
        "attn.out.bias": "attn_output.bias",
        # only in gpt-oss
        "attn.qkv.weight": "attn.wqkv.weight",
        "attn.qkv.bias": "attn.wqkv.bias",
        "attn.sinks": "attn_sinks",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.gate.bias": "ffn_gate_inp.bias",
        # block RMSNorm
        "mlp.norm.scale": "ffn_norm_scale.weight",
    }

    def dequant_mxfp4(
        blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16
    ) -> torch.Tensor:
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        # unpack nibbles
        lo = blocks & 0x0F
        hi = blocks >> 4
        loaded_blocks = torch.stack((lo, hi), dim=-1)
        loaded_blocks = loaded_blocks.view(
            *loaded_blocks.shape[:-2], loaded_blocks.shape[-2] * 2
        )
        # unbiased exponent
        loaded_scales = scales.int() - 127  # (..., 1)

        fp4_values = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
        loaded_tensor = torch.ldexp(
            fp4_values[loaded_blocks.int()], loaded_scales.unsqueeze(-1)
        )
        loaded_tensor = loaded_tensor.view(*loaded_tensor.shape[:-2], -1)
        return loaded_tensor

    def expert_key(layer: int, field: str) -> str:
        return f"blk.{layer}.{field}"

    tensors: dict[str, torch.Tensor] = {}
    for key in baseMapping:
        tensors[key] = st.get_tensor(baseMapping[key])

    layers: dict[int, dict[str, torch.Tensor]] = {}
    for key in st.keys():
        parts = key.split(".", 2)
        if parts[0] != "block":
            continue
        layer = int(parts[1])
        if layer not in layers:
            layers[layer] = {}
        layers[layer][parts[2]] = st.get_tensor(key)

    for layer, weights in layers.items():
        # 1-to-1 copy from attnMapping
        for ck_src, ten in weights.items():
            if ck_src in attnMapping:

                tensors[f"blk.{layer}.{attnMapping[ck_src]}"] = ten

        # dequantize experts in moe
        block1 = weights["mlp.mlp1_weight.blocks"]
        scale1 = weights["mlp.mlp1_weight.scales"]
        weight1 = dequant_mxfp4(block1, scale1)  # (E, 5760, 2880)
        D = weight1.shape[1] // 2

        tensors[expert_key(layer, "ffn_gate_exps.weight")] = weight1[:, :D, :]  # gate
        tensors[expert_key(layer, "ffn_up_exps.weight")] = weight1[:, D:, :]  # up
        tensors[expert_key(layer, "ffn_gate_exps.bias")] = weights["mlp.mlp1_bias"][
            :, :D
        ]
        tensors[expert_key(layer, "ffn_up_exps.bias")] = weights["mlp.mlp1_bias"][:, D:]

        block2 = weights["mlp.mlp2_weight.blocks"]
        scale2 = weights["mlp.mlp2_weight.scales"]
        if not args.quantize_moe:
            weight2 = dequant_mxfp4(block2, scale2)  # (E, 2880, 2880)
        tensors[expert_key(layer, "ffn_down_exps.weight")] = weight2
        tensors[expert_key(layer, "ffn_down_exps.bias")] = weights["mlp.mlp2_bias"]

    props = hp.to_gguf_props()

    # Add any metadata parameters that start with "_"
    config_json = dataclasses.asdict(hp)
    meta_params = {k: v for k, v in config_json.items() if k.startswith("_")}
    props.update(meta_params)

    tensors = [
        DefaultPrimitiveTensor(name=name, data=tensors[name]) for name in tensors.keys()
    ]
    theta = Theta(tensors)

    dataset = Dataset(props, theta)
    dataset.save(irpa_path, io_report_callback=logger.info)
