from brevitas.graph.quantize import quantize as brevitas_quantize
from brevitas.quant.scaled_int import (
    Int8WeightPerTensorFloat,
    Int8WeightPerChannelFloat,
    Int32Bias,
)
from brevitas.quant.experimental.float_quant_fnuz import (
    Fp8e4m3FNUZActPerTensorFloat,
)
from brevitas import nn as qnn
from torch import nn
from sharktank import ops
from sharktank.types.tensors import unbox_tensor

# Import SHARK layer types
from ..layers import (
    LinearLayer, 
    Conv2DLayer, 
    RMSNormLayer, 
    LayerNorm, 
    TokenEmbeddingLayer,
    QuantizationLayer
)

# Default SHARK to Brevitas compute layer mappings
SHARK_COMPUTE_LAYER_MAP = {
    LinearLayer: (qnn.QuantLinear, {
        'in_features': lambda module: module.weight.shape[1],
        'out_features': lambda module: module.weight.shape[0],
        'bias': lambda module: module.bias is not None,
        'weight_quant': Int8WeightPerTensorFloat,
        'bias_quant': Int32Bias,
        'return_quant_tensor': True
    }),
    Conv2DLayer: (qnn.QuantConv2d, {
        'weight_quant': Int8WeightPerChannelFloat,
        'bias_quant': Int32Bias,
        'return_quant_tensor': True
    }),
    TokenEmbeddingLayer: (qnn.QuantEmbedding, {
        'weight_quant': Int8WeightPerTensorFloat,
        'return_quant_tensor': True
    }),
    # Note: LayerNorm and RMSNorm don't have direct Brevitas equivalents
    # They would typically use QuantScaleBias for their weight/bias parameters
    LayerNorm: (qnn.QuantScaleBias, {
        'weight_quant': Int8WeightPerTensorFloat,
        'bias_quant': Int32Bias,
        'return_quant_tensor': True
    }),
    RMSNormLayer: (qnn.QuantScaleBias, {
        'weight_quant': Int8WeightPerTensorFloat,
        'return_quant_tensor': True
    }),
    QuantizationLayer: (qnn.QuantIdentity, {
        'act_quant': Fp8e4m3FNUZActPerTensorFloat,
        'return_quant_tensor': True
    }),
}

# SHARK activation quantization mappings (empty for now)
SHARK_QUANT_ACT_MAP = {}

# Activations that should use unsigned quantization (empty for now)
SHARK_UNSIGNED_ACT_TUPLE = ()

def quantize(
        model,
        compute_layer_map=SHARK_COMPUTE_LAYER_MAP,
        quant_act_map=SHARK_QUANT_ACT_MAP,
        unsigned_act_tuple=SHARK_UNSIGNED_ACT_TUPLE,
        requantize_layer_handler_output=True):
    """Quantize a PyTorch model using Brevitas with SHARK layer mappings.
    
    Args:
        model: PyTorch model to quantize (will be FX traced automatically)
        compute_layer_map: Mapping of SHARK compute layers to Brevitas quantized equivalents
        quant_act_map: Mapping of activation functions to quantized equivalents  
        unsigned_act_tuple: Tuple of activations that should use unsigned quantization
        requantize_layer_handler_output: Whether to requantize layer handler outputs
        
    Returns:
        Quantized model
    """
    # Auto-trace the model if it's not already traced, treating SHARK layers as leaf modules
    if not hasattr(model, 'graph'):
        import torch.fx
        
        # Get all SHARK layer types from our compute layer map
        shark_layer_types = tuple(compute_layer_map.keys())
        
        class SHARKLeafTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                return isinstance(m, shark_layer_types)
        
        tracer = SHARKLeafTracer()
        model = torch.fx.GraphModule(model, tracer.trace(model))
    
    return brevitas_quantize(
        model,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        unsigned_act_tuple=unsigned_act_tuple,
        requantize_layer_handler_output=requantize_layer_handler_output
    )