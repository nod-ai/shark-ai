# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Callable

from torch import Tensor
from ._registry import *
from sharktank.types.tensors import ReplicatedTensor, QuantizedTensor
from sharktank.types.quantizers import QuantizerTensor

from .signatures import *

import iree.turbine.ops.iree


@replicate.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def replicate_quantized(
    tensor: QuantizedTensor | QuantizerTensor, *, count: int, devices: tuple[int, ...]
) -> ReplicatedTensor:
    assert count == len(devices)
    return ReplicatedTensor(ts=tensor, shard_count=count, devices=devices)


@transfer_to_logical_device.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def transfer_to_logical_device_planar_quantized_tensor(
    tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.transfer_to_logical_device, tensor, ordinal
    )


@barrier_on_logical_device.override(AnyOfType(QuantizedTensor, QuantizerTensor))
def barrier_on_logical_device_planar_quantized_tensor(
    tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    return transfer_or_barrier(
        iree.turbine.ops.iree.barrier_on_logical_device, tensor, ordinal
    )


def transfer_or_barrier(
    operation: Callable, tensor: QuantizedTensor | QuantizerTensor, ordinal: int
):
    def operation_transform(globals: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: operation(f"{ordinal}", v) for k, v in globals.items()}

    return tensor.transform_subtensors(
        operation_transform, copy_external_tensor_trait=False
    )
