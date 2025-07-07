# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


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


@transfer_to_logical_device.override(
    TensorQuantizedWithLayoutType(BlockScaledFp4Layout)
)
def transfer_to_logical_device_default(tensor: PlanarQuantizedTensor, ordinal: int):
    layout_old: BlockScaledFp4Layout = tensor.layout
    _d = iree.turbine.ops.iree.transfer_to_logical_device(
        f"{ordinal}", tensor.layout._d
    )
    _qs = iree.turbine.ops.iree.transfer_to_logical_device(
        f"{ordinal}", layout_old.qs_bit_packed
    )
    layout_new = BlockScaledFp4Layout(
        layout_old.shape,
        _d,
        _qs,
        block_size=layout_old.block_size,
        use_fe8m0_scale=layout_old.use_fe8m0_scale,
    )
    return PlanarQuantizedTensor(
        shape=tensor.shape,
        layout=layout_new,
        name=tensor.name,
    )
