# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import random

from amdsharktank.models.llama.testing import quantize_theta_to_fp4
from amdsharktank.models.llama import toy_llama
from amdsharktank.types import (
    DefaultPrimitiveTensor,
    DynamicFp4BlockQuantizer,
    InferenceTensor,
    QuantizerTensor,
    unbox_tensor,
)
from amdsharktank.utils.testing import assert_tensor_close


def test_fp4_quantized_toy_llama(deterministic_random_seed):
    """Test that a roundtrip through quantization results in the same model."""

    theta, config = toy_llama.generate2(seed=0, dtype_rest=torch.float32)

    def unbox_transform(t: InferenceTensor) -> DefaultPrimitiveTensor | list:
        if isinstance(t, QuantizerTensor):
            return []
        return DefaultPrimitiveTensor(name=t.name, data=unbox_tensor(t))

    def remove_quantizers(t: InferenceTensor) -> DefaultPrimitiveTensor | list:
        if isinstance(t, QuantizerTensor):
            return []
        return t

    quantized_theta = quantize_theta_to_fp4(
        theta,
        quantizer=DynamicFp4BlockQuantizer(block_size=4, use_amdsharktank_kernel=False),
    )
    dequantized_theta = quantized_theta.transform(unbox_transform)

    quantized_theta_without_quantizers = quantized_theta.transform(remove_quantizers)

    assert_tensor_close(
        dequantized_theta.flatten(),
        quantized_theta_without_quantizers.flatten(),
        rtol=0,
        atol=0,
    )
