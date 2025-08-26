# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import math
from typing import Any, Dict, Iterable, Optional, List, Tuple
import torch
from sharktank.types import ShardedTensor, InferenceTensor
from sharktank.utils import longest_equal_range


def _reshape_infer_dynamic_dim(
    shape1: List[int], shape2: List[int]
) -> Tuple[List[int], List[int]]:
    assert (
        len([d for d in list(shape1) + list(shape2) if d < 0]) <= 1
    ), "Only one dynamic dimension is allowed"
    shape1_dynamic_dims = [i for i, d in enumerate(shape1) if d <= 0]
    if len(shape1_dynamic_dims) > 0:
        s2, s1 = _reshape_infer_dynamic_dim(shape2, shape1)
        return s1, s2

    shape2_dynamic_dims = [i for i, d in enumerate(shape2) if d <= 0]
    if len(shape2_dynamic_dims) == 0:
        assert math.prod(shape1) == math.prod(
            shape2
        ), f"Size mismatch: {shape1} vs {shape2}"
        return shape1, shape2

    shape2_dynamic_dim = shape2_dynamic_dims[0]
    shape1_size = math.prod(shape1)
    shape2_size_without_dynamic_dim = math.prod(d for d in shape2 if d > 0)
    shape2_res = list(shape2)
    assert shape1_size % shape2_size_without_dynamic_dim == 0
    shape2_res[shape2_dynamic_dim] = shape1_size // shape2_size_without_dynamic_dim
    assert shape2_res[shape2_dynamic_dim] > 0
    return shape1, shape2_res


def _reshape_get_single_split_dim(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would split a single dimension, return its index and the length of the new dimensions.
    If the reshape is not of that kind return `None`.
    E.g.
    _reshape_get_single_split_dim(from_shape=(2, 12, 5), to_shape=(2, 3, 4, 5))
    results in
    (1, 2)"""
    from_shape, to_shape = _reshape_infer_dynamic_dim(from_shape, to_shape)

    if len(to_shape) < len(from_shape):
        return None
    i = longest_equal_range(from_shape, to_shape)
    split_dims_length = len(to_shape) - len(from_shape) + 1
    if i == len(from_shape):
        return (
            i,
            split_dims_length,
        )
    j = len(to_shape) - longest_equal_range(reversed(from_shape), reversed(to_shape))
    assert i < j
    expected_split_dim_size = math.prod(to_shape[i:j])
    if expected_split_dim_size == 1:
        # 1's were inserted.
        return (
            i,
            split_dims_length,
        )
    if expected_split_dim_size != from_shape[i]:
        return None
    return (
        i,
        split_dims_length,
    )


def _reshape_get_flatten_dim_range(
    from_shape: List[int], to_shape: List[int]
) -> Optional[Tuple[int, int]]:
    """If a reshape would flatten a range of dimensions return that index range [begin, end).
    If the reshape is not of that kind return `None`."""
    flatten_start_len = _reshape_get_single_split_dim(to_shape, from_shape)
    if flatten_start_len is None:
        return None
    start, length = flatten_start_len
    return start, start + length


def assert_on_same_devices(*tensors: Tuple[ShardedTensor]) -> None:
    """
    Checks that all tensors are placed on the same devices.
    """
    if len(tensors) <= 1:
        return
    assert all(isinstance(tensor, ShardedTensor) for tensor in tensors)

    for tensor in tensors[1:]:
        if any(d0 != d for d0, d in zip(tensors[0].devices, tensor.devices)):
            raise ValueError("All tensors must be placed on the same devices.")


def transfer_n_pin(f):
    """
    Wrapper for each NON-TRANSFERRING op defined in this file.
    """

    def func_wrapper(*args: Tuple, **kwargs: Dict[str, Any]):
        """
        Wraps each NON-TRANSFERRING operation, f, to ensure that all incoming tensors are on the same device and that the result has the devices correctly labelled.

        If no ShardedTensors are present in the input, then no changes are made to input/output.
        """
        sharded_tensors = []
        for value in itertools.chain(args, kwargs.values()):
            if isinstance(value, ShardedTensor):
                sharded_tensors.append(value)
                continue
            if isinstance(
                value,
                (
                    InferenceTensor,
                    torch.Tensor,
                ),
            ):
                continue
            if isinstance(value, Iterable):
                for val in value:
                    if isinstance(val, ShardedTensor):
                        sharded_tensors.append(val)

        assert_on_same_devices(*sharded_tensors)
        res = f(*args, **kwargs)
        if len(sharded_tensors) > 0:
            if isinstance(res, ShardedTensor):
                res = res.clone(devices=sharded_tensors[0].devices)
            elif isinstance(res, Iterable) and all(
                isinstance(r, ShardedTensor) for r in res
            ):
                res = type(res)(
                    r.clone(devices=sharded_tensors[0].devices) for r in res
                )
        return res

    func_wrapper._impl_name = getattr(f, "_impl_name", None)  # For impl selection
    return func_wrapper


def wrap_override(signature_dispatcher_override):
    """
    Wrap [op].override's result so that the transfer_n_pin(f) becomes the target in _TargetOverride rather than f itself.
    """

    def override_return_wrapper(*override_args, **override_kwargs):
        orig_decorator = signature_dispatcher_override(
            *override_args, **override_kwargs
        )
        new_decorator = lambda f: orig_decorator(transfer_n_pin(f))
        return new_decorator

    return override_return_wrapper
