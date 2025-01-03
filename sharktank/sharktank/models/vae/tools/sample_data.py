# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Various utilities for deriving stable sample data for the model."""

from pathlib import Path

import torch


def get_random_inputs(dtype, device, bs: int = 2):
    height = 1024
    width = 1024
    return torch.rand(bs, 4, width // 8, height // 8, dtype=dtype).to(device)
