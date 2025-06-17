# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from brevitas.graph.quantize import layerwise_quantize as brevitas_quantize
from brevitas.quant.scaled_int import (
    Int32Bias,
)
from brevitas import nn as qnn
from torch import nn

# Import SHARK layer types
from ..layers import LinearLayer, QuantizationLayer

# Default SHARK to Brevitas compute layer mappings
SHARK_COMPUTE_LAYER_MAP = {
    LinearLayer: (
        qnn.QuantLinear,
        {
            "in_features": lambda module: module.weight.shape[1],
            "out_features": lambda module: module.weight.shape[0],
            "bias": lambda module: module.bias is not None,
            "input_quant": lambda module: module.quantization_config.quantization_scheme,
            "weight_quant": lambda module: module.quantization_config.quantization_scheme,
            "bias_quant": Int32Bias,
            "cache_inference_quant_bias": True,
            "return_quant_tensor": False,
        },
    ),
    QuantizationLayer: (
        qnn.QuantIdentity,
        {
            "act_quant": lambda module: module.quantization_config.quantization_scheme,
            "return_quant_tensor": False,
        },
    ),
}

# SHARK activation quantization mappings (empty for now)
SHARK_QUANT_ACT_MAP = {}

# Activations that should use unsigned quantization (empty for now)
SHARK_UNSIGNED_ACT_TUPLE = ()


def quantize(
    model,
    compute_layer_map=SHARK_COMPUTE_LAYER_MAP,
    name_blacklist=None,
):
    """Quantize a PyTorch model using Brevitas with SHARK layer mappings.

    Args:
        model: PyTorch model to quantize
        compute_layer_map: Mapping of SHARK compute layers to Brevitas quantized equivalents
        name_blacklist: Optional blacklist of module names to skip during quantization

    Returns:
        Quantized model
    """
    return brevitas_quantize(
        model,
        compute_layer_map=compute_layer_map,
        name_blacklist=name_blacklist,
    )
