# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

This module implements the QuaRot quantization scheme based on the paper:
"QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (arxiv:2404.00456)
https://arxiv.org/pdf/2404.00456

QuaRot uses Hadamard rotations to remove outliers from activations and weights,
enabling effective 4-bit quantization without significant accuracy loss.
"""

from typing import Optional
import torch
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
from sharktank.layers.base import ThetaLayer


def apply_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Apply Hadamard transform using recursive implementation.

    Args:
        x: Input tensor, last dimension must be power of 2
        normalize: Whether to normalize by 1/sqrt(k)

    Returns:
        Hadamard transformed tensor
    """
    k = x.shape[-1]

    if k & (k - 1) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {k}")

    def generate_hadamard_matrix(n, dtype, device):
        if n == 1:
            return torch.tensor([[1.0]], dtype=dtype, device=device)
        else:
            h_n_half = generate_hadamard_matrix(n // 2, dtype, device)
            top = torch.cat([h_n_half, h_n_half], dim=1)
            bottom = torch.cat([h_n_half, -h_n_half], dim=1)
            return torch.cat([top, bottom], dim=0)

    H = generate_hadamard_matrix(k, x.dtype, x.device)

    orig_shape = x.shape
    x_flat = x.view(-1, k)
    # TODO: There may be ways to speed this up, but the performance impact is most likely quite small currently.
    result = torch.matmul(x_flat, H.T)

    if normalize:
        result = result / (k**0.5)

    return result.view(orig_shape)


class QuaRotTransform(ThetaLayer):
    """QuaRot Hadamard transform with random sign flips and 4-bit quantization."""

    def __init__(
        self, theta, hidden_dim: int, seed: Optional[int] = None, block_size: int = 32
    ):
        """Initialize QuaRot transform.

        Args:
            theta: Theta containing quarot_signs or None for random generation
            hidden_dim: Hidden dimension (must be power of 2)
            seed: Random seed for reproducibility
            block_size: Block size for 4-bit quantization
        """
        super().__init__(theta)
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.block_size = block_size

        quarot_signs = self.theta.optional_tensor("quarot_signs")
        if quarot_signs is not None:
            self.signs = quarot_signs.as_torch()
        else:
            self.signs = None

        # TODO: Look into variant quantization methods like online/symmetric
        # as described in the QuaRot paper instead of FP4 block quantization
        self.quantizer = DynamicFp4BlockQuantizer(block_size=block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard transform with optional randomization."""
        if self.signs is not None:
            signs = self.signs.to(x.device)
            x_signed = x * signs
            return apply_hadamard_transform(x_signed)
        else:
            return apply_hadamard_transform(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse Hadamard transform with optional randomization."""
        x_transformed = apply_hadamard_transform(x)
        if self.signs is not None:
            signs = self.signs.to(x.device)
            return x_transformed * signs
        else:
            return x_transformed
