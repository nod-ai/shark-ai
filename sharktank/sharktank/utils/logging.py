# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch

from iree.turbine.support.logging import get_logger


transform_logger: logging.Logger = get_logger("amdsharktank.transforms")


def format_tensor_statistics(tensor: torch.Tensor):
    return f"mean = {tensor.mean()}, median = {tensor.median()}, std dev = {tensor.std()}, min = {tensor.min()}, max = {tensor.max()}"
