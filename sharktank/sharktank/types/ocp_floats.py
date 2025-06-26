# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""OCP (Open Compute Project) floating point format configurations and utilities.

Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

This module provides standard OCP floating point format definitions and configurations
for quantization operations. It supports various FP4 formats like E2M1, E3M2, etc.
"""

from enum import Enum
from typing import Dict, Tuple

import torch

__all__ = [
    "FloatingPointFormat",
    "FloatingPointConfig",
    "get_fp4_lookup_table",
    "generate_fp4_lookup_table",
    "convert_fp4_scales_to_float",
    "compute_fp4_block_scales",
    "fp4_e2m1_to_float32",
    "float32_to_fp4_e2m1",
    "e8m0_to_float32",
    "float32_to_e8m0",
]


class FloatingPointFormat(Enum):
    """OCP (Open Compute Project) based floating point formats."""

    E2M1 = "e2m1"
    E3M2 = "e3m2"
    E2M3 = "e2m3"


class FloatingPointConfig:
    """Configuration for FP4 formats with configurable parameters."""

    def __init__(
        self,
        exponent_bits: int,  # eX
        mantissa_bits: int,  # mX
        exponent_bias: int,
        has_inf: bool = False,  # f
        has_nan: bool = False,  # n
        has_unsigned_zero: bool = False,  # uz
    ):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.exponent_bias = exponent_bias
        self.has_inf = has_inf
        self.has_nan = has_nan
        self.has_unsigned_zero = has_unsigned_zero
        self.total_bits = 1 + exponent_bits + mantissa_bits  # sign + exp + mantissa


# OCP FP4 format configurations
_FP4_CONFIGS: Dict[FloatingPointFormat, FloatingPointConfig] = {
    FloatingPointFormat.E2M1: FloatingPointConfig(
        exponent_bits=2,
        mantissa_bits=1,
        exponent_bias=1,
        has_inf=False,
        has_nan=False,
        has_unsigned_zero=True,
    ),
}


def generate_fp4_lookup_table(fmt: FloatingPointFormat) -> torch.Tensor:
    """Generate lookup table for FP4 format.

    Args:
        fmt: The floating point format to generate a lookup table for

    Returns:
        torch.Tensor: Lookup table mapping FP4 indices to float32 values

    Raises:
        ValueError: If the format is not supported
    """
    # TODO: Process f/n/uz arguments
    if fmt not in _FP4_CONFIGS:
        raise ValueError(f"Unsupported floating point format: {fmt}")

    config = _FP4_CONFIGS[fmt]
    num_values = 2**config.total_bits
    values = []

    for i in range(num_values):
        # Extract bits: sign (1 bit) | exponent (E bits) | mantissa (M bits)
        sign_bit = (i >> (config.exponent_bits + config.mantissa_bits)) & 1
        exp_bits = (i >> config.mantissa_bits) & ((1 << config.exponent_bits) - 1)
        mant_bits = i & ((1 << config.mantissa_bits) - 1)

        # Calculate the value
        implicit_leading_bit = 0.0 if exp_bits == 0 else 1.0
        effective_exponent = (1 if exp_bits == 0 else exp_bits) - config.exponent_bias
        mantissa = implicit_leading_bit + mant_bits / (1 << config.mantissa_bits)
        value = (2**effective_exponent) * mantissa

        # Apply sign
        if sign_bit:
            value = -value

        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


# Hardcoded FP4 E2M1 lookup table for performance (indices 0-15 -> float32 values)
_FP4_E2M1_TO_FP32 = torch.tensor(
    [
        0.0,  # 0000: zero
        0.5,  # 0001: 2^(-1) * 1.0
        1.0,  # 0010: 2^0 * 1.0
        1.5,  # 0011: 2^0 * 1.5
        2.0,  # 0100: 2^1 * 1.0
        3.0,  # 0101: 2^1 * 1.5
        4.0,  # 0110: 2^2 * 1.0
        6.0,  # 0111: 2^2 * 1.5
        -0.0,  # 1000: negative zero
        -0.5,  # 1001: - 2^(-1) * 1.0
        -1.0,  # 1010: - 2^0 * 1.0
        -1.5,  # 1011: - 2^0 * 1.5
        -2.0,  # 1100: - 2^1 * 1.0
        -3.0,  # 1101: - 2^1 * 1.5
        -4.0,  # 1110: - 2^2 * 1.0
        -6.0,  # 1111: - 2^2 * 1.5
    ],
    dtype=torch.float32,
)

# Pre-computed lookup tables for supported FP4 formats
_FP4_LOOKUP_TABLES: Dict[FloatingPointFormat, torch.Tensor] = {
    FloatingPointFormat.E2M1: _FP4_E2M1_TO_FP32,
}


def get_fp4_lookup_table(fmt: FloatingPointFormat) -> torch.Tensor:
    """Get the lookup table for a specific FP4 format.

    Args:
        fmt: The floating point format

    Returns:
        torch.Tensor: Lookup table mapping FP4 indices to float32 values

    Raises:
        ValueError: If the format is not supported
    """
    if fmt not in _FP4_LOOKUP_TABLES:
        raise ValueError(f"No lookup table available for format: {fmt}")

    return _FP4_LOOKUP_TABLES[fmt]


"""Floating Point Utilities"""

# FP4 constants
_FP4_E2M1_MIN_VALUE = -6.0
_FP4_E2M1_MAX_VALUE = 6.0
_FP4_MIN_INDEX = 0
_FP4_MAX_INDEX = 15


def e8m0_to_float32(e8m0_values: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 (8 exponent bits, 0 mantissa bits) values to float32.

    E8M0 format uses IEEE-style bias of 127. The value is computed as:
    2^(e8m0_value - 127)

    Args:
        e8m0_values: Tensor of uint8 values representing e8m0 exponents

    Returns:
        torch.Tensor: Corresponding float32 values
    """
    return torch.pow(2.0, e8m0_values.float() - 127.0)


def float32_to_e8m0(values: torch.Tensor) -> torch.Tensor:
    """Convert float32 values to e8m0 (8 exponent bits, 0 mantissa bits) format.

    E8M0 format uses IEEE-style bias of 127. The e8m0 value is computed as:
    log2(value) + 127

    Args:
        values: Tensor of positive float32 values

    Returns:
        torch.Tensor: Corresponding uint8 e8m0 values, clamped to [0, 255]
    """
    return torch.log2(values).add(127.0).clamp(0, 255).to(torch.uint8)


def convert_fp4_scales_to_float(
    scales: torch.Tensor, use_fe8m0_scale: bool
) -> torch.Tensor:
    if use_fe8m0_scale:
        return e8m0_to_float32(scales)
    else:
        return scales


def compute_fp4_block_scales(
    block_max: torch.Tensor,
    use_fe8m0_scale: bool,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute FP4 block scales from block maximum values.

    Args:
        block_max: Maximum absolute values per block [num_blocks, 1]
        use_fe8m0_scale: Whether to use FE8M0 scales
        dtype: Data type for epsilon calculation

    Returns:
        Tuple of (scales, scales_float) where scales are in storage format
        and scales_float are ready for computation
    """
    if use_fe8m0_scale:
        finfo = torch.finfo(dtype)
        block_max.clamp_(min=finfo.eps)
        fe8m0_scales = torch.ceil(block_max)
        e8m0_values = float32_to_e8m0(fe8m0_scales)
        scales = e8m0_values.squeeze(-1)
        scales_float = e8m0_to_float32(e8m0_values)
    else:
        # Use regular float scales - scale to use full FP4 range
        finfo = torch.finfo(torch.float32)
        scales_float = block_max / _FP4_E2M1_MAX_VALUE
        scales_float.clamp_(min=finfo.eps)  # In-place clamp
        scales = scales_float.squeeze(-1)

    return scales, scales_float


def fp4_e2m1_to_float32(fp4_indices: torch.Tensor) -> torch.Tensor:
    """Convert FP4 E2M1 format indices to float32 values using lookup table.

    Args:
        fp4_indices: Tensor containing FP4 indices in range [0, 15] as unpacked uint8

    Returns:
        torch.Tensor: Corresponding float32 values

    Raises:
        ValueError: If indices are outside the valid range [0, 15]
    """
    if torch.any(fp4_indices < _FP4_MIN_INDEX) or torch.any(
        fp4_indices > _FP4_MAX_INDEX
    ):
        raise ValueError(
            f"FP4 indices must be in range [{_FP4_MIN_INDEX}, {_FP4_MAX_INDEX}], got min={fp4_indices.min().item()}, max={fp4_indices.max().item()}"
        )

    lookup_table = get_fp4_lookup_table(FloatingPointFormat.E2M1)
    return lookup_table[fp4_indices.long()]


def float32_to_fp4_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Convert float32 values to FP4 E2M1 format indices via quantization.

    Finds the closest FP4 E2M1 representation for each input value by computing
    absolute differences with all possible FP4 values and selecting the minimum.

    Args:
        values: Input tensor of float32 values to quantize

    Returns:
        torch.Tensor: FP4 indices as unpacked uint8 values in range [0, 15]
    """
    lookup_table = get_fp4_lookup_table(FloatingPointFormat.E2M1)

    # Find closest FP4 value for each input
    values_expanded = values.unsqueeze(-1)  # [..., 1]
    lookup_expanded = lookup_table.unsqueeze(0).expand(*values.shape, -1)  # [..., 16]

    # Compute absolute differences and find minimum
    abs_diff = torch.abs(values_expanded - lookup_expanded)
    fp4_indices = torch.argmin(abs_diff, dim=-1)

    return fp4_indices.to(torch.uint8)
