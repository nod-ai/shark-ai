# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union, Optional
from numbers import Number
import torch


def ceildiv(a: int | float, b: int | float) -> int | float:
    return -(a // -b)


def round_up_to_multiple_of(x: Number, multiple: Number) -> Number:
    return x + (-x % multiple)


def cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, /, *, dim: Optional[Union[int, tuple[int]]] = None
) -> torch.Tensor:
    """Compute cosine similarity over dimensions dim.
    If dim is none computes over all dimensions."""
    dot_product = torch.sum(a * b, dim=dim)
    norm_a = a.pow(2).sum(dim=dim).sqrt()
    norm_b = b.pow(2).sum(dim=dim).sqrt()
    return dot_product / (norm_a * norm_b)
