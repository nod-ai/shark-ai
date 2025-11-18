# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import torch

from typing import Callable, Sequence, Any, Dict, List, Optional, Tuple
from amdsharktank.utils import tree
from amdsharktank.types import (
    DefaultPrimitiveTensor,
    PrimitiveTensor,
    QuantizedTensor,
    QuantizedLayout,
)
from amdsharktank.types.tensors import unbox_tensor
from ._registry import SignatureDispatcher, AnyType, _matches

__all__ = [
    "promote_to_float",
    "trivially_replicable",
    "get_all_implementations",
    "cast_to_type_spec",
]


def _is_layout_type(t) -> bool:
    """Check if t is a QuantizedLayout subclass (not instance)."""
    return isinstance(t, type) and issubclass(t, QuantizedLayout)


def promote_to_float(tensor: torch.Tensor) -> torch.Tensor:
    """Promote to an appropriate floating point dtype that would result in "acceptable"
    loss of precision."""
    if tensor.dtype.is_floating_point:
        return tensor

    if tensor.dtype.itemsize <= 4:
        return tensor.to(dtype=torch.float32)

    return tensor.to(dtype=torch.float64)


def call_trivially_replicable(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Call a function with replicated tensor arguments that can be called with
    unsharded tensors.

    It is expected that all replicated tensors have the same number of shards.
    The function will be called with each set of matching shard indices and
    replicated tensor(s) will be constructed from the resulting shards.

    This wrapper handles tree-like structures in the arguments and results. It will
    traverse them and handle nested tensors. Note that sequence and dictionary types
    will be reconstructed with dict and tuple types, so the underlying function fn
    must be polymorphic in this regard and not expect concrete types.

    When collecting the results all non-tensor results are taken from the first device.
    """

    from amdsharktank.types import (
        ReplicatedTensor,
        is_any_tensor,
    )

    flat_args = tree.flatten(
        (
            args,
            kwargs,
        ),
        is_leaf=_is_leaf,
    )
    first_replicated_tensor_arg = [
        arg for arg in flat_args if isinstance(arg, ReplicatedTensor)
    ]
    first_replicated_tensor_arg = (
        None
        if len(first_replicated_tensor_arg) == 0
        else first_replicated_tensor_arg[0]
    )
    if first_replicated_tensor_arg is None:
        # No replicated tensor arguments, just call the function.
        return fn(*args, **kwargs)

    shard_count = first_replicated_tensor_arg.shard_count

    per_shard_args_and_kwargs = [
        _extract_per_shard_args_and_kwargs(i, args, kwargs) for i in range(shard_count)
    ]
    per_shard_results = [
        fn(*per_shard_args, **per_shard_kwargs)
        for per_shard_args, per_shard_kwargs in per_shard_args_and_kwargs
    ]

    def reduce_to_list_of_tensors_fn(a: Any, b: Any) -> Any:
        if not is_any_tensor(b):
            return a
        a.append(b)
        return a

    def tensor_to_empty_list_if_tensor(x: Any) -> Any:
        return [] if is_any_tensor(x) else x

    reduce_initial = tree.map_leaves(
        per_shard_results[0], f=tensor_to_empty_list_if_tensor, is_leaf=_is_leaf
    )

    # Make a lists of tensor shards that correspond to the same replicated tensor.
    reduced_results = tree.reduce_horizontal(
        fn=reduce_to_list_of_tensors_fn,
        trees=per_shard_results,
        is_leaf=_is_leaf,
        initial=reduce_initial,
    )

    def make_replicated_tensor_if_tensor_collection(x: Any) -> Any:
        if _is_tensor_collection(x):
            return ReplicatedTensor(ts=x, devices=first_replicated_tensor_arg.devices)
        return x

    result_with_replicated_tensor = tree.map_leaves(
        reduced_results,
        f=make_replicated_tensor_if_tensor_collection,
        is_leaf=_is_leaf_or_tensor_collection,
    )
    return result_with_replicated_tensor


def trivially_replicable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator to turn an trivially replicable operation into a replicated
    operation.

    An operation is trivially replicable if the replicated variant just executes the
    operation on all devices with the respective shards.

    E.g.
    ```
    @trivially_replicable
    def fn(a: torch.Tensor) -> torch.Tensor:
        return a

    arg = torch.Tensor([1, 2, 3])
    shard_count = 2
    replicated_arg = ReplicatedTensor(
        ts=arg, shard_count=shard_count
    )
    fn(replicated_arg)
    ```

    ```
    ReplicatedTensor(<unnamed>, [3], shard_count=2 of [3])
    ```
    """

    def wrapper(*args, **kwargs):
        return call_trivially_replicable(fn, args, kwargs)

    return wrapper


def _extract_per_shard_args_and_kwargs(
    shard_index: int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return tree.map_leaves(
        (args, kwargs),
        f=functools.partial(
            _extract_tensor_shard_if_replicate, shard_index=shard_index
        ),
        is_leaf=_is_leaf,
    )


def _extract_tensor_shard_if_replicate(v: Any, shard_index: int) -> Any:
    from amdsharktank.types import (
        ReplicatedTensor,
    )

    if isinstance(v, ReplicatedTensor):
        return v.shards[shard_index]
    return v


def _is_leaf(v: Any) -> bool:
    from amdsharktank.types import is_any_tensor

    return is_any_tensor(v) or tree.is_leaf_default(v)


def _is_leaf_or_tensor_collection(x: Any) -> bool:
    return _is_leaf(x) or _is_tensor_collection(x)


def _is_tensor_collection(x: Any) -> bool:
    from amdsharktank.types import is_any_tensor

    return isinstance(x, Sequence) and all(is_any_tensor(v) for v in x)


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


def _cast_single_input(
    input_value, expected_type, layout_to_quantizer=None, layout_type=None
):
    """Cast a single input to match the expected type."""
    if input_value is None or expected_type is AnyType:
        return input_value

    if _matches(expected_type, torch.Tensor):
        return unbox_tensor(input_value)

    if _matches(expected_type, PrimitiveTensor):
        return DefaultPrimitiveTensor(data=unbox_tensor(input_value))

    # Check if expected_type is a QuantizedLayout subclass or QuantizedTensor
    if _is_layout_type(expected_type) or _matches(expected_type, QuantizedTensor):
        if not layout_to_quantizer:
            raise ValueError(
                f"No layout_to_quantizer mapping provided; cannot automatically cast to {expected_type}."
            )

        if expected_type not in layout_to_quantizer:
            raise ValueError(
                f"{expected_type} not in {layout_to_quantizer}; cannot automatically cast."
            )
        quantizer_fn = layout_to_quantizer[expected_type]
        quantizer = quantizer_fn(input_value.dtype)
        quantized_tensor = quantizer.quantize(input_value)

        if _is_layout_type(expected_type):
            return quantized_tensor.unpack()
        else:
            return quantized_tensor

    return input_value


def cast_to_type_spec(
    inputs: List[Any],
    type_spec: Tuple[type, ...],
    layout_to_quantizer: Optional[Dict[str, Callable]] = None,
    layout_types: Optional[Tuple[type, ...]] = None,
) -> List[Any]:
    """Cast inputs to match the type specification.

    Args:
        inputs: List of input values (tensors or None)
        type_spec: Tuple of expected types from the override
        layout_to_quantizer: Optional mapping from layout types to quantizer functions
        layout_types: Optional tuple of layout types corresponding to each input

    Returns:
        List of inputs cast to appropriate types
    """
    result = []

    for i, input_value in enumerate(inputs):
        if i >= len(type_spec):
            result.append(input_value)
        else:
            layout_type = (
                layout_types[i] if layout_types and i < len(layout_types) else None
            )
            result.append(
                _cast_single_input(
                    input_value, type_spec[i], layout_to_quantizer, layout_type
                )
            )

    return result
