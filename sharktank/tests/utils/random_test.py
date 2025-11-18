# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch

from amdsharktank.utils.random import (
    make_simple_calculable_weight_torch,
    make_wide_range_weights,
)


def test_make_wide_range_weights_reproducible_and_scaled():
    shape = [128, 64]
    weights_a = make_wide_range_weights(shape, dtype=torch.float32)
    weights_b = make_wide_range_weights(shape, dtype=torch.float32)

    assert torch.allclose(weights_a, weights_b)
    assert weights_a.shape == torch.Size(shape)
    assert weights_a.dtype == torch.float32

    fan_in = shape[-1]
    expected_std = 0.8 / math.sqrt(fan_in)
    sample_std = weights_a.std(unbiased=False).item()
    assert abs(sample_std - expected_std) / expected_std < 0.05


def test_make_simple_calculable_weight_torch_pattern():
    shape = [2, 3, 5]
    weights = make_simple_calculable_weight_torch(shape, dtype=torch.float32)

    expected_pattern = [0.0, 1.0, -1.0, 0.5, 2.0]
    flattened = weights.flatten().tolist()

    assert weights.shape == torch.Size(shape)
    assert weights.dtype == torch.float32
    for idx, value in enumerate(flattened):
        assert value == expected_pattern[idx % len(expected_pattern)]
