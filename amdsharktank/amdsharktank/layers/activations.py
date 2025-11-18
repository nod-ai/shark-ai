# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from torch import nn
from amdsharktank import ops

# TODO: don't use nn.functional directly.
ACT2FN = {
    "gelu": nn.functional.gelu,
    "gelu_new": ops.gelu_tanh_approximation,
    "relu": nn.functional.relu,
    "quick_gelu": ops.gelu_sigmoid_approximation,
    "silu": nn.functional.silu,
    "swiglu": ops.swiglu,
}
