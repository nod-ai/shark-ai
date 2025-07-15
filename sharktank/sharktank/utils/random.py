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


# Range of torch.rand() is [0,1)
# Range of torch.rand() * 2 - 1 is [-1, 1), includes negative values
def make_rand_torch(shape: list[int], dtype: Optional[torch.dtype] = torch.float32):
    return (torch.rand(shape) * 2 - 1).to(dtype=dtype)


def make_random_mask(shape: tuple[int], dtype: Optional[torch.dtype] = None):
    mask = make_rand_torch(shape=shape, dtype=dtype)
    mask = (mask >= 0).to(dtype=dtype)
    return mask


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
