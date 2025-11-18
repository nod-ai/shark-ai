# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Various utilities for deriving stable sample data for the model."""

import math
import torch
from amdsharktank.utils.logging import get_logger

logger = get_logger(__name__)


def get_random_inputs(
    *,
    dtype: torch.dtype,
    device: torch.device,
    bs: int = 2,
    config: str = "sdxl",
    width: int = 1024,
    height: int = 1024,
    latent_channels: int = 16,
):
    if config == "sdxl":
        logger.debug("sdxl returning inputs")
        assert width % 8 == 0 and height % 8 == 0
        return torch.rand(bs, 4, width // 8, height // 8, dtype=dtype).to(device)
    elif config == "flux":
        logger.debug("flux returning inputs")
        return torch.rand(
            bs,
            math.ceil(height / 16) * math.ceil(width / 16),
            4 * latent_channels,
            dtype=dtype,
        ).to(device)
    else:
        logger.debug("config: ", config)
        raise AssertionError(f"{config} config not implmented [sdxl, flux] implemented")
