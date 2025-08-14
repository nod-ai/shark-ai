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


def _cast_single_input(input_value, expected_type, layout_to_quantizer=None):
    """Cast a single input to match the expected type."""
    from torch import Tensor

    if input_value is None or expected_type is AnyType:
        return input_value

    if _matches(expected_type, Tensor):
        if isinstance(input_value, PrimitiveTensor):
            return unbox_tensor(input_value)
        return input_value

    if _matches(expected_type, PrimitiveTensor):
        if isinstance(input_value, Tensor) and not isinstance(
            input_value, PrimitiveTensor
        ):
            return DefaultPrimitiveTensor(data=input_value)
        return input_value

    if _matches(expected_type, QuantizedTensor):
        if isinstance(input_value, QuantizedTensor):
            return input_value
        if isinstance(input_value, Tensor) and layout_to_quantizer:
            # Use TensorScaledLayout as default for quantized tensors in tests
            from sharktank.types.layouts import TensorScaledLayout

            if TensorScaledLayout in layout_to_quantizer:
                quantizer_fn = layout_to_quantizer[TensorScaledLayout]
                quantizer = quantizer_fn(input_value.dtype)
                return quantizer.quantize(input_value)
        return input_value

    # Unknown type, pass through
    return input_value


def cast_to_type_spec(
    inputs: List[Any],
    type_spec: Tuple[type, ...],
    layout_to_quantizer: Optional[Dict[str, Callable]] = None,
) -> List[Any]:
    """Cast inputs to match the type specification.

    Args:
        inputs: List of input values (tensors or None)
        type_spec: Tuple of expected types from the override
        layout_to_quantizer: Optional mapping from layout names to quantizer functions

    Returns:
        List of inputs cast to appropriate types
    """
    result = []

    for i, input_value in enumerate(inputs):
        if i >= len(type_spec):
            # Past the end of type_spec, just pass through
            result.append(input_value)
        else:
            result.append(
                _cast_single_input(input_value, type_spec[i], layout_to_quantizer)
            )

    return result
