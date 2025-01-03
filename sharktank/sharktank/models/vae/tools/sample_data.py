# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Various utilities for deriving stable sample data for the model."""

from pathlib import Path

import torch


def get_random_inputs(dtype, device, bs: int = 2, config: str = "sdxl"):
    height = 1024
    width = 1024
    if config == "sdxl":
        print("sdxl returning inputs")
        return torch.rand(bs, 4, width // 8, height // 8, dtype=dtype).to(device)
    elif config == "flux":
        print("flux returning inputs")
        return torch.rand(bs, int(width * height / 256), 64, dtype=dtype).to(device)
    else:
        print("config: ", config)
        raise AssertionError(f"{config} config not implmented [sdxl, flux] implemented")
