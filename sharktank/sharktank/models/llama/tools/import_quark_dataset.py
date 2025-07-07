# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Imports quark pre-processed weights and quantization config into a
Dataset of the gguf format.

Usage:
  python -m sharktank.models.llama.tools.import_quark_dataset \
    --params=llama2-7b-fp8.safetensors --output-irpa-file=new.irpa \
    --config-json=../llama2/config.json

"""
from typing import Optional

from safetensors.torch import save_file
import json
from pathlib import Path
import safetensors
import sys
import torch

from sharktank.types import *
from sharktank.types.tensors import serialized_name_to_dtype
from sharktank.layers.configs.llm_configs import (
    _int_prop,
    _float_prop,
    _optional_int_prop,
    _int_prop,
)


def _load_json(p: Path):
    print(f"Loading {p}")
    with open(p, "rb") as f:
        return json.load(f)


def _get_dataset_props(config_json_struct) -> dict:
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json_struct.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json_struct.items() if not k.startswith("_")}
    return {
        "meta": meta_params,
        "hparams": hparams,
    }


def _load_theta(st_source) -> Theta:
    tensors = [
        DefaultPrimitiveTensor(name=name, data=st_source.get_tensor(name))
        for name in st_source.keys()
    ]
    return Theta(tensors)


def as_torch_or_none(tensor: Optional[InferenceTensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.as_torch()


def detect_quantization_format(
    weight_tensor: torch.Tensor, scale_tensor: torch.Tensor, block_size: int = 32
) -> str:
    """Detect whether weights are FP4 or FP8 based on tensor shapes."""
    # TODO: Expand on this to support other dtypes including mxfp6 quantization
    weight_elements = weight_tensor.numel()
    scale_elements = scale_tensor.numel()

    # For FP4: 2 values packed per byte, so weight_elements * 2 / block_size should equal scale_elements
    expected_fp4_blocks = (weight_elements * 2 + block_size - 1) // block_size

    # For FP8: 1 value per byte, so weight_elements / block_size should equal scale_elements
    expected_fp8_blocks = (weight_elements + block_size - 1) // block_size

    if abs(scale_elements - expected_fp4_blocks) < abs(
        scale_elements - expected_fp8_blocks
    ):
        return "fp4"
    else:
        return "fp8"


def infer_original_tensor_shape(
    weight_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    block_size: int,
    format: str,
) -> list[int]:
    """Infer original tensor shape from quantized weight and scale tensors."""
    if format == "fp4":
        # FP4: 2 values packed per byte
        total_elements = weight_tensor.numel() * 2
    else:
        # FP8: 1 value per byte
        total_elements = weight_tensor.numel()

    # Use scale tensor shape to help infer dimensions
    scale_shape = scale_tensor.shape
    if len(scale_shape) == 2:
        # Assume [output_dim, input_dim // block_size]
        output_dim = scale_shape[0]
        input_dim = scale_shape[1] * block_size
        return [output_dim, input_dim]
    elif len(scale_shape) == 1:
        # Linear layer with single dimension
        return [scale_shape[0] * block_size]
    else:
        raise ValueError(f"Unsupported scale tensor shape: {scale_shape}")


def create_fp4_block_quantizer(
    weight_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    layer_name: str,
    block_size: int = 32,
) -> "PlanarQuantizedTensor":
    """Create StaticFp4BlockQuantizer from Quark FP4 weights and scales."""

    # Convert U8 scales to appropriate format
    # Quark uses E8M0 format for FP4 scales
    # Keep the original 2D shape instead of flattening
    scales = scale_tensor.to(torch.float32)

    # Infer original tensor shape
    original_shape = infer_original_tensor_shape(
        weight_tensor, scale_tensor, block_size, "fp4"
    )

    # Create the FP4 block layout directly
    layout = BlockScaledFp4Layout(
        shape=original_shape,
        d=scales,
        qs=weight_tensor,  # Already packed FP4 data
        block_size=block_size,
        use_fe8m0_scale=True,
    )

    quantized_tensor = PlanarQuantizedTensor(
        shape=original_shape,
        name=layer_name,
        layout=layout,
    )

    return quantized_tensor


def hf_to_gguf(layer_name: str) -> str:
    assert layer_name.startswith("model.layers")
    mapping = {
        "input_layernorm": "attn_norm",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "post_attention_layernorm": "ffn_norm",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }

    # Split the input string
    parts = layer_name.split(".")

    # Extract the numerical value and the key to be mapped
    numerical_value = parts[2]  # The part after "models.layers" and its number
    key_to_map = ".".join(parts[3:])

    # Map the key
    if key_to_map in mapping:
        mapped_value = mapping[key_to_map]
    else:
        raise ValueError(f"Mapping for '{key_to_map}' not found.")

    # Construct the output string
    output_str = f"blk.{numerical_value}.{mapped_value}"
    return output_str


def apply_per_layer_quant(
    root_theta: Theta,
    layer_name: str,
    updated_tensors: dict[str, InferenceTensor],
    n_head: int,
    split_sizes: list[int],
    block_size: int = 32,
    weight_dtype_override: Optional[torch.dtype] = None,
):
    """Take the quantization parameters and hf weights from the imported Theta
    and create InferenceTensors out of them, converting their names to gguf format
    in the process.
    """
    layer_theta = root_theta(layer_name)
    weight_quant_scale = layer_theta.tensor("weight_scale").as_torch()
    if weight_quant_scale.dtype == torch.bfloat16:
        weight_quant_scale = weight_quant_scale.to(torch.float32)
    weight = layer_theta.tensor("weight").as_torch()

    quant_format = detect_quantization_format(weight, weight_quant_scale, block_size)

    if quant_format == "fp4":
        quantized_weight = weight
    else:
        # It looks dumb but, this step is required for numerical correctness against quark.
        # weight = weight.view(torch.float8_e4m3fn)
        weight = (weight.to(torch.float64) * weight_quant_scale).to(torch.float32)
        quantized_weight = None  # Will be created in quantize_weight function

    weight_quant_zero_point = layer_theta.optional_tensor("weight_zero_point")
    if weight_quant_zero_point == None:
        weight_quant_zero_point = torch.zeros(1, dtype=torch.float32)
    else:
        weight_quant_zero_point = weight_quant_zero_point.as_torch()
    input_quant_scale = as_torch_or_none(layer_theta.optional_tensor("input_scale"))
    if input_quant_scale is not None and input_quant_scale.dtype is torch.bfloat16:
        input_quant_scale = input_quant_scale.to(torch.float32)
    output_quant_scale = as_torch_or_none(layer_theta.optional_tensor("output_scale"))
    if output_quant_scale and output_quant_scale.dtype is torch.bfloat16:
        output_quant_scale = output_quant_scale.to(torch.float32)
    if output_quant_scale is not None and weight_dtype_override is not None:
        output_quant_scale = output_quant_scale.to(weight_dtype_override)
    if weight_quant_scale is None:
        print("weight quant scale not found for layer ", layer_name)
        return

    layer_parent = ".".join(layer_name.split(".")[:-1])

    def quantize_weight(
        weight_name: str,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: Optional[torch.Tensor],
    ):
        if quant_format == "fp4":
            fp4_tensor = create_fp4_block_quantizer(
                quantized_weight, weight_scale, weight_name, block_size
            )
            updated_tensors[weight_name] = fp4_tensor
        else:
            # Our scale is the reciprocal of the quark scale
            # We multiply scale by two to account for diff between fnuz and fn
            weight_quantizer = StaticScaledQuantizer(
                scale=1.0 / (weight_scale * 2.0),
                reciprocal_scale=(weight_scale * 2.0),
                offset=None
                if (weight_zp is None or torch.count_nonzero(weight_zp) == 0)
                else weight_zp,
                dtype=torch.float8_e4m3fnuz,
            )
            weight_quant = weight_quantizer.quantize(weight, name=weight_name)
            updated_tensors[weight_quant.name] = weight_quant

    # In older quark models the qkv layer is fused. Unfuse.
    if "qkv" in layer_name:
        # The qkv layer is fused in the quark model, decompose back into individual q, k , and v weights
        q_weight, k_weight, v_weight = torch.split(weight, split_sizes)
        q_weight = (
            q_weight.reshape(
                n_head, 2, q_weight.shape[0] // n_head // 2, *q_weight.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(q_weight.shape)
        )
        k_weight = (
            k_weight.reshape(
                n_head, 2, k_weight.shape[0] // n_head // 2, *k_weight.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(k_weight.shape)
        )
        q_name = hf_to_gguf(layer_parent + ".q_proj")
        k_name = hf_to_gguf(layer_parent + ".k_proj")
        v_name = hf_to_gguf(layer_parent + ".v_proj")
        quantize_weight(
            q_name + ".weight", q_weight, weight_quant_scale, weight_quant_zero_point
        )
        quantize_weight(
            k_name + ".weight", k_weight, weight_quant_scale, weight_quant_zero_point
        )
        quantize_weight(
            v_name + ".weight", v_weight, weight_quant_scale, weight_quant_zero_point
        )
        # The output and input quantizers are duplicated for each of the q, k, and v weights
        names = [f"{i}.qdq_output" for i in [q_name, k_name, v_name]]
        for name in names:
            updated_tensors[name] = StaticScaledQuantizer(
                name=name,
                scale=1.0 / (output_quant_scale * 2.0),
                reciprocal_scale=output_quant_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
            )
        names = [f"{i}.q_input" for i in [q_name, k_name, v_name]]
        for name in names:
            updated_tensors[name] = StaticScaledQuantizer(
                name=name,
                scale=1.0 / (input_quant_scale * 2.0),
                reciprocal_scale=input_quant_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
            )
        # Remove the updated tensors from the original tree.
        root_theta.pop(layer_parent + ".q_proj")
        root_theta.pop(layer_parent + ".k_proj")
        root_theta.pop(layer_parent + ".v_proj")
        root_theta.pop(layer_name)

    else:
        new_layer_name = hf_to_gguf(layer_name)
        quantize_weight(
            new_layer_name + ".weight",
            weight,
            weight_quant_scale,
            weight_quant_zero_point,
        )
        # we explicitly provide the reciprocal scale because converting from float16 to float8 after doing 1/scale results in significant numerical differences
        # scales are multipled by two to account for the difference between fnuz and fn
        if input_quant_scale is not None:
            updated_tensors[new_layer_name + ".q_input"] = StaticScaledQuantizer(
                name=new_layer_name + ".q_input",
                scale=1.0 / (input_quant_scale * 2.0),
                reciprocal_scale=input_quant_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
            )
        if output_quant_scale is not None:
            updated_tensors[new_layer_name + ".q_output"] = StaticScaledQuantizer(
                name=new_layer_name + ".q_output",
                scale=1.0 / (output_quant_scale * 2.0),
                reciprocal_scale=output_quant_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
            )

        # Remove the updated tensor from the original tree.
        root_theta.pop(layer_name)


def convert_hf_hparams_to_gguf(hf_hparams: dict[str, any]) -> dict[str, any]:
    hp = hf_hparams["hparams"]
    attention_head_count = _int_prop(hp, "num_attention_heads")
    attn_head_dim = int(
        _int_prop(hp, "hidden_size") // _int_prop(hp, "num_attention_heads")
    )
    attn_head_dim = int(_optional_int_prop(hp, "head_dim", attn_head_dim))

    return {
        "llama.context_length": _int_prop(hp, "max_position_embeddings"),
        "llama.embedding_length": _int_prop(hp, "hidden_size"),
        "llama.block_count": _int_prop(hp, "num_hidden_layers"),
        "llama.feed_forward_length": _int_prop(hp, "intermediate_size"),
        "llama.rope.dimension_count": attn_head_dim,
        "llama.attention.head_count": attention_head_count,
        "llama.attention.layer_norm_rms_epsilon": _float_prop(hp, "rms_norm_eps"),
        "llama.attention.head_count_kv": _optional_int_prop(
            hp, "num_key_value_heads", attention_head_count
        ),
    }


def update_norm_layer(
    quant_theta: Theta,
    layer_name: str,
    updated_tensors: dict[str, InferenceTensor],
    weight_dtype_override: Optional[torch.dtype] = None,
):
    """Convert layernames for non quantized tensors and add them to the updated_tensors dict"""
    for sub in ["input_layernorm", "post_attention_layernorm"]:
        sub_name = layer_name + "." + sub
        new_name = hf_to_gguf(sub_name) + ".weight"
        single_replace(
            quant_theta, sub_name, new_name, updated_tensors, weight_dtype_override
        )

    if "self_attn" in quant_theta(layer_name).keys:
        layer_idx = layer_name.split(".")[-1]
        if "kv_cache_scale" in quant_theta(layer_name, "self_attn").keys:
            kv_cache_scale = (
                quant_theta(layer_name, "self_attn")
                .tensor("kv_cache_scale")
                .as_torch()
                .to(torch.float32)
            )
            if weight_dtype_override is not None:
                kv_cache_scale = kv_cache_scale.to(weight_dtype_override)
            new_name = f"blk.{layer_idx}.kv_cache"
            updated_tensors[new_name] = StaticScaledQuantizer(
                name=new_name + ".quantizer",
                scale=1.0 / (kv_cache_scale * 2.0),
                reciprocal_scale=kv_cache_scale * 2.0,
                dtype=torch.float8_e4m3fnuz,
            )

        if "prob_output_scale" in quant_theta(layer_name, "self_attn").keys:
            prob_output_scale = (
                quant_theta(layer_name, "self_attn")
                .tensor("prob_output_scale")
                .as_torch()
                .to(torch.float32)
                * 2.0
            )
            if weight_dtype_override is not None:
                prob_output_scale = prob_output_scale.to(weight_dtype_override)
            new_name = f"blk.{layer_idx}.attn_scale"
            updated_tensors[new_name] = DefaultPrimitiveTensor(
                name=new_name, data=prob_output_scale
            )
    else:
        assert False


def single_replace(
    quant_theta: Theta,
    layer_name: str,
    gguf_name: str,
    updated_tensors: dict[str, InferenceTensor],
    dtype_override: Optional[torch.dtype] = None,
):
    data = quant_theta(layer_name).tensor("weight").as_torch()
    if dtype_override is not None and data.dtype != dtype_override:
        data = data.to(dtype_override)
    updated_tensors[gguf_name] = DefaultPrimitiveTensor(name=gguf_name, data=data)


def main(argv):
    from sharktank.utils import cli

    parser = cli.create_parser()
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--config-json", type=Path, required=True, help="Path to the config.json file"
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.safetensors"),
        help="Parameter file name, relative to config.json",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="7b",
        help="Base model to use for split sizes to decompose the qkv tensor. Default is 7b, 70b is also supported.",
        choices=["7b", "70b", "405b"],
    )
    parser.add_argument(
        "--fp4-block-size",
        type=int,
        default=32,
        help="Block size for FP4 quantization (default: 32)",
    )
    parser.add_argument(
        "--fp4-scale-format",
        choices=["fe8m0", "float"],
        default="fe8m0",
        help="Scale format for FP4 quantization (default: fe8m0)",
    )
    parser.add_argument(
        "--weight-dtype-override",
        type=str,
        default=None,
        help="Data type to cast output_scale and certain weights to (e.g., float32, float16, bfloat16)",
    )
    args = cli.parse(parser, args=argv)

    config_json_path: Path = args.config_json
    params_path: Path = args.params
    # TODO: find a way to get this programatically so we don't have to flag for it
    split_sizes = [4096, 4096, 4096] if args.model_base == "7b" else [8192, 1024, 1024]
    layers_per_base = {"7b": 32, "70b": 40, "405b": 125}
    num_layers = layers_per_base[args.model_base]
    # Construct the pre-transform dataset.
    dataset_props = _get_dataset_props(_load_json(config_json_path))
    OVERRIDE_OUTSCALE_NAME = False
    overridden = []
    with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
        quant_theta = _load_theta(st)
        OVERRIDE_OUTSCALE_NAME = "prob_output_scale" not in [
            key.split(".")[-1] for key in st.keys()
        ]
        if OVERRIDE_OUTSCALE_NAME:
            overridden = [
                key for key in st.keys() if "output_scale" in key.split(".")[-1]
            ]
    updates = {}
    if OVERRIDE_OUTSCALE_NAME:
        print(
            "WARNING: overriding name of k and v output scales to make them kv_cache_quantizer scales instead of linear output quantizers"
        )
        for o in overridden:
            theta_name = o
            # theta_name looks like: model.layers.9.self_attn.k_proj.output_scale
            # we want: model.layers.9.self_attn
            prefix = ".".join(theta_name.split(".")[:-2])
            value = quant_theta.tensor(o)
            # remove unwanted theta subtree
            quant_theta.pop(o)
            new_name = prefix + ".kv_cache_scale"
            updates[new_name] = value
        for key, value in quant_theta.flatten().items():
            updates[key] = value
        quant_theta = Theta(updates)

    ds = Dataset(dataset_props, quant_theta)

    # Convert hyperparams to gguf format
    updated_properties = convert_hf_hparams_to_gguf(ds.properties)

    head_count = (updated_properties["llama.attention.head_count"],)

    updated_tensors: dict[str, InferenceTensor] = {}
    model_layers = [f"model.layers.{i}" for i in range(num_layers)]

    # Convert weight_dtype_override string to torch dtype
    weight_dtype_override = (
        serialized_name_to_dtype(args.weight_dtype_override)
        if args.weight_dtype_override
        else None
    )

    sub_layers = [
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ]
    for layer in model_layers:
        for sub in sub_layers:
            layer_name = layer + "." + sub
            apply_per_layer_quant(
                quant_theta,
                layer_name,
                updated_tensors,
                n_head=head_count[0],
                split_sizes=split_sizes,
                block_size=args.fp4_block_size,
                weight_dtype_override=weight_dtype_override,
            )

    # Update the non quantized weights (norm layers)
    for layer_idx in model_layers:
        update_norm_layer(
            quant_theta,
            layer_idx,
            updated_tensors,
            weight_dtype_override,
        )

    # The stragglers
    stragglers = [
        ("model.embed_tokens", "token_embd.weight"),
        ("model.norm", "output_norm.weight"),
        ("lm_head", "output.weight"),
    ]
    for layer, new_name in stragglers:
        single_replace(
            quant_theta, layer, new_name, updated_tensors, weight_dtype_override
        )

    new_theta = Theta(updated_tensors)
    # Make a new Dataset from the updated properties and tensors.
    new_ds = Dataset(updated_properties, new_theta)
    new_ds.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main(sys.argv[1:])
