# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple, Dict, Any, Iterable
import itertools
from sharktank.types import ShardedTensor, InferenceTensor
import torch


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
