# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Callable

from torch import Tensor
from sharktank.types.layouts import BlockScaledFp4Layout
from ._registry import *
from sharktank.types.tensors import (
    PlanarQuantizedTensor,
    ReplicatedTensor,
    QuantizedTensor,
)
from .signatures import *

import iree.turbine.ops.iree


@replicate.override(QuantizedTensor)
def replicate_quantized(
    input: QuantizedTensor, *, count: int, devices: tuple[int, ...]
) -> ReplicatedTensor:
    assert count == len(devices)
    return ReplicatedTensor(ts=input, shard_count=count, devices=devices)


@transfer_to_logical_device.override(QuantizedTensor)
def transfer_to_logical_device_planar_quantized_tensor(
    tensor: QuantizedTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.transfer_to_logical_device, tensor, ordinal
    )


@barrier_on_logical_device.override(QuantizedTensor)
def barrier_on_logical_device__planar_quantized_tensor(
    tensor: QuantizedTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.barrier_on_logical_device, tensor, ordinal
    )


def transfer_or_barrier(operation: Callable, tensor: QuantizedTensor, ordinal: int):
    def operation_transform(globals: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: operation(f"{ordinal}", v) for k, v in globals.items()}

    return tensor.transform_globals(operation_transform)
