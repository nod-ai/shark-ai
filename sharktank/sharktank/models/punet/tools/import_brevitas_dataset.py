# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Imports Brevitas pre-processed weights and quantization config into a
Dataset.

Usage:
  python -m amdsharktank.models.punet.import_hf_dataset \
    --output-irpa-file ~/models/punet/punet_fp16.irpa \
    --config-json ~/models/stable-diffusion-xl-base-1.0/unet/config.json

The resulting dataset has all tensors as nested in the original model.
Properties are separated into a "meta" dict (for "_" prefixed props) and an
"hparams" dict.

Default flag values assume that there is a quant_param.json and
params.safetensors adjacent to the HF config.json file.
"""
from typing import Optional

import json
from pathlib import Path
import safetensors
import sys
import torch

from amdsharktank.types import *
from amdsharktank.utils import cli

# It is possible to import quant params from stock unet weights for testing.
# Quality won't be great but needs SMOOTHQUANT prescaling disabled to work
# at all.
IMPORT_SMOOTHQUANT_PRESCALE = True


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


def _dtype_str_to_dtype(dtype_str: str) -> torch.dtype:
    prefix = "torch."
    assert dtype_str.startswith(prefix), f"Expected 'torch.`DTYPE`', got {dtype_str}"
    return getattr(torch, dtype_str[len(prefix) :])


def apply_per_layer_quant(
    root_theta: Theta, layer_name: str, qp, updated_tensors: dict[str, InferenceTensor]
):
    # The config file has tensors as 1d and a _shape suffixed array with the
    # concrete shape.
    def _get_json_tensor(
        name: str, dtype: Optional[torch.dtype]
    ) -> Optional[torch.Tensor]:
        data_1d = qp.get(name)
        if data_1d is None:
            return None
        if dtype is not None:
            assert (
                f"{name}_dtype" not in qp
            ), f"Explicit dtype for {name} but json has dtype"
        else:
            # _dtype key must come from the dict
            dtype_str = qp[f"{name}_dtype"]
            dtype = _dtype_str_to_dtype(dtype_str)

        shape = qp[f"{name}_shape"]
        return torch.tensor(data_1d, dtype=dtype).reshape(shape)

    # If the quantization layer is for the output softmax we only have
    # the activation scale. This will likely change when the softmax
    # quantization carries the full type and scale.
    if (
        layer_name.endswith("output_softmax_quant")
        or layer_name.endswith("out_q")
        or layer_name.endswith("out_k")
        or layer_name.endswith("out_v")
    ):
        output_scale = _get_json_tensor("act_scale", torch.float32)
        dtype = _dtype_str_to_dtype(qp.get("act_zp_dtype"))
        output_quantizer = StaticScaledQuantizer(
            name=layer_name,
            scale=1.0 / output_scale,
            reciprocal_scale=output_scale,
            dtype=dtype,
        )

        updated_tensors[output_quantizer.name] = output_quantizer
        return

    layer_theta = root_theta(layer_name)
    weight = layer_theta.tensor("weight")
    weight_dtype = weight.as_torch().dtype
    bias = layer_theta.optional_tensor("bias")

    # Optional activation pre-multiplier.
    if IMPORT_SMOOTHQUANT_PRESCALE:
        smoothquant_mul = _get_json_tensor("smoothquant_mul", dtype=weight_dtype)
        if smoothquant_mul is not None:
            premul_input = DefaultPrimitiveTensor(
                name=f"{layer_name}.premul_input",
                data=smoothquant_mul,
            )
            updated_tensors[premul_input.name] = premul_input

    input_scale = _get_json_tensor("input_scale", torch.float32)
    output_scale = _get_json_tensor("output_scale", torch.float32)
    weight_scale = _get_json_tensor("weight_scale", torch.float32)
    weight_zp = _get_json_tensor("weight_zp", dtype=None)

    # In the current version, we assume that the input is per-tensor quantized
    # for signed arithmetic.
    input_zp = _get_json_tensor("input_zp", dtype=None)
    if input_zp is not None:
        assert torch.count_nonzero(input_zp.to(torch.float32)) == 0

    # Currently, there seems to be no standardization in `quant_params.json` for fields in every layer
    # across different quantization schemes (int8, fp8). int8 quantization was the first end-to-end tested
    # quantization scheme so there's some defaults to that.
    quantization_type = (
        qp.get("input_zp_dtype")
        if qp.get("input_zp_dtype") is not None
        else "torch.int8"
    )
    quantization_dtype = tensors.serialized_name_to_dtype(
        quantization_type.split(".")[-1]
    )
    if output_scale is not None:
        output_quantizer = StaticScaledQuantizer(
            name=f"{layer_name}.q_output",
            scale=1.0 / output_scale,
            reciprocal_scale=output_scale,
            dtype=torch.float8_e4m3fnuz,  # hardcoded for right now until breviatas updates
        )
        updated_tensors[output_quantizer.name] = output_quantizer

    if (
        input_scale is None
        and input_zp is None
        and weight_scale is None
        and weight_zp is None
    ):
        # Non quantized layer that has had possible adjustments above.
        return

    # Quantized layer must have all quantization info.
    assert (
        weight_scale is not None
    ), f"Could not find weight scale (in {qp.keys()}) for {layer_name}"
    assert (
        input_scale is not None
    ), f"Could not find input scale (in {qp.keys()}) for {layer_name}"

    def quantize_weight(
        weight_name: str,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: Optional[torch.Tensor],
    ):
        # Weight scaling.
        # There is an implicit assumption that the weight is asym (uint8) quantized.
        # Our quantizer uses scale/offset nomenclature. The offset maps to
        # zero-point, and the scale maps to the *dequant* scale (so terms differ
        # by reciprocal).
        weight_quantizer = StaticScaledQuantizer(
            scale=1.0 / weight_scale,
            reciprocal_scale=weight_scale,
            offset=(
                None
                if (weight_zp is None or torch.count_nonzero(weight_zp) == 0)
                else weight_zp
            ),
            dtype=quantization_dtype,
        )
        weight_quant = weight_quantizer.quantize(weight, name=weight_name)
        updated_tensors[weight_quant.name] = weight_quant

    def quantize_bias(
        bias_name: str,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        # If the bias is present, it dictates the overall output quantization
        # and will not be checked for correct parameters at runtime. It must
        # be quantized to match properly.
        bias_scale = 1.0 / (input_scale * weight_scale)
        bias_quantizer = StaticScaledQuantizer(
            scale=bias_scale,
            dtype=torch.int32 if quantization_dtype == torch.int8 else torch.float16,
            disable_saturate=True,
        )
        bias_quant = bias_quantizer.quantize(bias, name=bias_name)
        updated_tensors[bias_quant.name] = bias_quant

    # If dealing with a fused QKV layer, then we need to split the weight as
    # Brevitas concats it along the [n * output_dim, input_dim] axis, where
    # n is 2 (to_kv) or 3 (to_qkv).
    # We do this here vs in the model since it is much more efficient to
    # operate on the weights contiguously at rest.
    # TODO: Maybe just have Brevitas output it this way vs having us need
    # to special case this so extensively (some special casing will be needed).
    if layer_name.endswith(".to_kv"):
        # MCHA fused layer.
        weight_k, weight_v = weight.chunk(2, dim=0)
        weight_scale_k, weight_scale_v = weight_scale.chunk(2, dim=0)
        if weight_zp is not None:
            weight_zp_k, weight_zp_v = weight_zp.chunk(2, dim=0)
        else:
            weight_zp_k = None
            weight_zp_v = None
        # print(
        #     f"Chunk MCHA KV into {weight_k.shape}, {weight_v.shape} from {weight.shape}"
        # )
        quantize_weight(
            f"{layer_name}.to_k.weight", weight_k, weight_scale_k, weight_zp_k
        )
        quantize_weight(
            f"{layer_name}.to_v.weight", weight_v, weight_scale_v, weight_zp_v
        )
        updated_tensors[weight.name] = None
        if bias is not None and quantization_dtype == torch.int8:
            bias_k, bias_v = bias.chunk(2, dim=0)
            quantize_bias(
                f"{layer_name}.to_k.bias", bias_k, input_scale, weight_scale_k
            )
            quantize_bias(
                f"{layer_name}.to_v.bias", bias_v, input_scale, weight_scale_v
            )
            updated_tensors[bias.name] = None
    elif layer_name.endswith(".to_qkv"):
        # MHA fused layer.
        weight_q, weight_k, weight_v = weight.chunk(3, dim=0)
        weight_scale_q, weight_scale_k, weight_scale_v = weight_scale.chunk(3, dim=0)
        if weight_zp is not None:
            weight_zp_q, weight_zp_k, weight_zp_v = weight_zp.chunk(3, dim=0)
        else:
            weight_zp_q = None
            weight_zp_k = None
            weight_zp_v = None
        # print(
        #     f"Chunk MHA QKV into {weight_q.shape}, {weight_k.shape}, {weight_v.shape} from {weight.shape}"
        # )
        quantize_weight(
            f"{layer_name}.to_q.weight", weight_q, weight_scale_q, weight_zp_q
        )
        quantize_weight(
            f"{layer_name}.to_k.weight", weight_k, weight_scale_k, weight_zp_k
        )
        quantize_weight(
            f"{layer_name}.to_v.weight", weight_v, weight_scale_v, weight_zp_v
        )
        updated_tensors[weight.name] = None
        if bias is not None and quantization_dtype == torch.int8:
            bias_q, bias_k, bias_v = bias.chunk(3, dim=0)
            quantize_bias(
                f"{layer_name}.to_q.bias", bias_q, input_scale, weight_scale_q
            )
            quantize_bias(
                f"{layer_name}.to_k.bias", bias_k, input_scale, weight_scale_k
            )
            quantize_bias(
                f"{layer_name}.to_v.bias", bias_v, input_scale, weight_scale_v
            )
            updated_tensors[bias.name] = None
    else:
        # Unfused.
        quantize_weight(weight.name, weight, weight_scale, weight_zp)
        if bias is not None:
            quantize_bias(bias.name, bias, input_scale, weight_scale)

    # Input scaling.
    # Assume per tensor scaling of input.
    assert len(input_scale.shape) == 0
    input_quantizer = StaticScaledQuantizer(
        name=f"{layer_name}.q_input",
        scale=1.0 / input_scale,
        reciprocal_scale=input_scale,
        dtype=quantization_dtype,
    )
    updated_tensors[input_quantizer.name] = input_quantizer


def main(argv):

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
        "--quant-params",
        type=Path,
        default=Path("quant_params.json"),
        help="Quantization parameters",
    )
    parser.add_argument(
        "--base-params",
        type=Path,
        help="Base parameters to initialize from (will be augmented with quantized)",
    )
    args = cli.parse(parser, args=argv)

    config_json_path: Path = args.config_json
    params_path: Path = args.params
    quant_params_path: Path = args.quant_params
    if not params_path.is_absolute():
        params_path = config_json_path.parent / params_path
    if not quant_params_path.is_absolute():
        quant_params_path = config_json_path.parent / quant_params_path

    # Construct the pre-transform dataset.
    dataset_props = _get_dataset_props(_load_json(config_json_path))
    quant_params_struct = _load_json(quant_params_path)
    with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
        quant_theta = _load_theta(st)
    base_theta = None
    if args.base_params is not None:
        print("Initializing from base parameters:", args.base_params)
        with safetensors.safe_open(
            args.base_params, framework="pt", device="cpu"
        ) as st:
            base_theta = _load_theta(st)

    ds = Dataset(dataset_props, quant_theta if base_theta is None else base_theta)

    # The quant_params_struct has quantization parameter structs keyed by full
    # layer name. We process each of these in turn to produce a per-layer
    # quantization scheme where no quantized tensors escape their layer.
    updated_tensors: dict[str, InferenceTensor] = {}
    for layer_name, qp in quant_params_struct.items():
        # print(f"Applying per-layer quants: {layer_name}")
        apply_per_layer_quant(quant_theta, layer_name, qp, updated_tensors)

    # Apply updates into a new Theta.
    theta = base_theta if base_theta is not None else quant_theta
    flat_tensors = theta.flatten()
    flat_tensors.update(updated_tensors)
    ds.root_theta = Theta(flat_tensors)

    # TODO: Post-process to introduce fused cross-layer connections.

    ds.save(args.output_irpa_file, io_report_callback=None)


if __name__ == "__main__":
    main(sys.argv[1:])
