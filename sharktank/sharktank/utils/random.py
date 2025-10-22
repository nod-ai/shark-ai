# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import random
import torch
import math


# Range of torch.rand() is [0,1)
# Range of torch.rand() * 2 - 1 is [-1, 1), includes negative values
def make_rand_torch(shape: list[int], dtype: Optional[torch.dtype] = torch.float32):
    return (torch.rand(shape) * 2 - 1).to(dtype=dtype)


def make_random_mask(shape: tuple[int], dtype: Optional[torch.dtype] = None):
    mask = make_rand_torch(shape=shape, dtype=dtype)
    mask = (mask >= 0).to(dtype=dtype)
    return mask


def make_wide_range_weights(
    shape: list[int], dtype: Optional[torch.dtype] = None, seed: int = 1234
) -> torch.Tensor:
    """Generate weights with proper variance scaling to prevent numerical explosions.

    Uses Xavier-like initialization: scale by 1/sqrt(fan_in) to keep output variance
    stable regardless of layer dimensions. The 0.8 factor provides diversity while
    maintaining numerical stability.

    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    fan_in = shape[-1]
    std = 0.8 / math.sqrt(fan_in)
    weights = torch.randn(shape, dtype=dtype, generator=generator) * std
    return weights


def make_simple_calculable_weight_torch(
    shape: list[int], dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Create simple weights that can be calculated by hand for analytical testing.
    """
    weights = torch.zeros(shape, dtype=dtype)
    flat_weights = weights.view(-1)

    # Simple pattern: 0, 1, -1, 0.5, 2, repeat...
    simple_values = [0.0, 1.0, -1.0, 0.5, 2.0]

    for i in range(flat_weights.numel()):
        flat_weights[i] = simple_values[i % len(simple_values)]

    return weights


@contextmanager
def fork_numpy_singleton_rng() -> Generator:
    """Fork the legacy Numpy RNG.
    This is meant to be used during testing to facilitate test isolation and determinism.
    Once Numpy's legacy singleton RNG is removed this should be removed."""
    orig_state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(orig_state)


@contextmanager
def fork_builtin_rng() -> Generator:
    orig_state = random.getstate()
    try:
        yield
    finally:
        random.setstate(orig_state)
