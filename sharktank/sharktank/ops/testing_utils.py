# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Testing utilities for ops."""

from typing import Callable, Dict, List, Any, Optional, Tuple
import torch

from sharktank.types import (
    DefaultPrimitiveTensor,
    PrimitiveTensor,
    QuantizedTensor,
)
from sharktank.types.tensors import unbox_tensor
from ._registry import SignatureDispatcher, AnyType, _matches


def get_all_implementations(op: SignatureDispatcher) -> Dict[str, Callable]:
    """Get all registered implementations for an op.

    Args:
        op: The op to get implementations for

    Returns:
        Dictionary mapping implementation names to callable functions
    """
    implementations = {}
    for override in op._overrides:
        impl_name = override.target.__name__
        implementations[impl_name] = override.target
    return implementations


def get_expected_layouts(func: Callable) -> Optional[Dict[str, type]]:
    """Extract expected layout types from a function decorated with @quantized_tensor_layout_of_type.

    Args:
        func: The function to extract layout expectations from

    Returns:
        Dictionary mapping parameter names to expected layout types, or None if not decorated
    """
    if hasattr(func, "_expected_layouts"):
        return func._expected_layouts
    return None


def get_override_type_spec(
    op: SignatureDispatcher, override_func: Callable
) -> Optional[Tuple[type, ...]]:
    """Extract the type specification for an override.

    Args:
        op: The op dispatcher
        override_func: The override function to get type spec for

    Returns:
        Tuple of types expected by the override, or None if not found
    """
    for override in op._overrides:
        if override.target == override_func:
            return override.type_spec
    return None


def cast_to_type_spec(
    inputs: List[Any],
    type_spec: Tuple[type, ...],
    override_func: Optional[Callable] = None,
    layout_to_quantizer: Optional[Dict[str, Callable]] = None,
) -> List[Any]:
    """Cast inputs to match the type specification.

    Args:
        inputs: List of input values (tensors or None)
        type_spec: Tuple of expected types from the override
        override_func: Optional override function to extract layout expectations from
        layout_to_quantizer: Optional mapping from layout names to quantizer functions

    Returns:
        List of inputs cast to appropriate types
    """
    from torch import Tensor
    import inspect

    result = []

    # Get expected layouts if available
    expected_layouts = None
    if override_func:
        expected_layouts = get_expected_layouts(override_func)

    # Get parameter names if we have an override_func
    param_names = None
    if override_func:
        sig = inspect.signature(override_func)
        param_names = list(sig.parameters.keys())

    # Handle case where we have more inputs than type_spec
    # (can happen with variable arguments)
    for i, input_value in enumerate(inputs):
        if i >= len(type_spec):
            # Past the end of type_spec, just pass through
            result.append(input_value)
            continue

        expected_type = type_spec[i]

        if input_value is None:
            result.append(None)
        elif expected_type is AnyType:
            # AnyType matches anything, pass through unchanged
            result.append(input_value)
        elif _matches(expected_type, Tensor):
            # Convert to plain tensor if needed
            if isinstance(input_value, Tensor):
                result.append(input_value)
            elif isinstance(input_value, PrimitiveTensor):
                result.append(unbox_tensor(input_value))
            else:
                result.append(input_value)
        elif _matches(expected_type, PrimitiveTensor):
            # Wrap in PrimitiveTensor if needed
            if isinstance(input_value, PrimitiveTensor):
                result.append(input_value)
            elif isinstance(input_value, Tensor):
                result.append(DefaultPrimitiveTensor(data=input_value))
            else:
                result.append(input_value)
        elif _matches(expected_type, QuantizedTensor):
            # Check if expected_type inherits from QuantizedTensor
            if isinstance(input_value, QuantizedTensor):
                result.append(input_value)
            elif isinstance(input_value, Tensor):
                # Check if we have a specific layout expectation
                if expected_layouts and param_names and i < len(param_names):
                    param_name = param_names[i]
                    if param_name in expected_layouts:
                        layout_type = expected_layouts[param_name]
                        if layout_to_quantizer and layout_type in layout_to_quantizer:
                            # Use the specific quantizer for this layout
                            quantizer_fn = layout_to_quantizer[layout_type]
                            # Use the same dtype as the input tensor for quantization
                            quantizer = quantizer_fn(input_value.dtype)
                            result.append(quantizer.quantize(input_value))
                            continue

                # No layout information available and no converter - this is an error
                raise ValueError(
                    f"Cannot convert tensor to {expected_type.__name__} without layout information. "
                    f"The implementation needs @quantized_tensor_layout_of_type decorator to specify expected layouts."
                )
            else:
                result.append(input_value)
        else:
            # Unknown type, pass through
            result.append(input_value)

    return result
