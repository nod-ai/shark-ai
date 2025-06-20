# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from sharktank.kernels import batched_block_scaled_mmt_fp4
from sharktank.types import BlockScaledFp4Layout
from sharktank.types import PlanarQuantizedTensor
from sharktank.types.layout_utils import pack_fp4_e2m1_to_uint8
from sharktank.types.ocp_floats import convert_fp4_scales_to_float
from sharktank.types.ocp_floats import fp4_e2m1_to_float32


def _reference_batched_block_scaled_mmt_fp4(
    a: torch.Tensor, d: torch.Tensor, qs_packed: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Reference implementation for batched block scaled FP4 MMT."""
    from sharktank.types.layout_utils import unpack_uint8_to_fp4_e2m1

    fp4_indices = unpack_uint8_to_fp4_e2m1(qs_packed)
    fp4_values = fp4_e2m1_to_float32(fp4_indices)

    # TODO: Support other dtypes better for scales
    common_dtype = torch.float32
    d_expanded = d.unsqueeze(-1).to(common_dtype)
    fp4_values = fp4_values.to(common_dtype)
    scaled_values = fp4_values * d_expanded

    # Reshape to [N, K] for matrix multiplication
    n, num_blocks, _ = scaled_values.shape
    k = num_blocks * block_size
    weight_matrix = scaled_values.reshape(n, k)

    # Perform batched matrix multiplication: a @ weight_matrix.T
    weight_matrix = weight_matrix.to(a.dtype)
    result = torch.bmm(a, weight_matrix.T.unsqueeze(0).expand(a.shape[0], -1, -1))

    return result


class TestBatchedBlockScaledMmtFp4:
    """Test class for batched block scaled MMT FP4 operations."""

    def _create_fp4_test_data(self, n: int, k: int, block_size: int, dtype: torch.dtype, use_power_of_two_scale: bool = False):
        """Helper method to create FP4 test data with consistent setup."""
        num_blocks = k // block_size
        
        if use_power_of_two_scale:
            # Create power-of-two scales (integer exponents)
            d = torch.randint(-8, 8, (n, num_blocks)).to(dtype)
        else:
            # Create regular float scales
            d = torch.randn(n, num_blocks, dtype=dtype)
        
        # Create FP4 indices
        fp4_indices = torch.randint(0, 16, (n, num_blocks, block_size), dtype=torch.uint8)
        qs_packed = pack_fp4_e2m1_to_uint8(fp4_indices)
        
        return d, fp4_indices, qs_packed

    def _create_planar_quantized_tensor(self, n: int, k: int, d: torch.Tensor, qs_packed: torch.Tensor, block_size: int, use_power_of_two_scale: bool = False):
        """Helper method to create PlanarQuantizedTensor with BlockScaledFp4Layout."""
        layout = BlockScaledFp4Layout(
            shape=[n, k],
            d=d,
            qs=qs_packed,
            block_size=block_size,
            use_power_of_two_scale=use_power_of_two_scale,
        )
        return PlanarQuantizedTensor(shape=[n, k], layout=layout)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("m", [32, 64])
    @pytest.mark.parametrize("n", [128, 256])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_batched_block_scaled_mmt_fp4(self, batch_size, m, n, k, block_size, dtype):
        """Test batched block scaled FP4 matrix multiplication."""
        torch.manual_seed(42)

        a = torch.randn(batch_size, m, k, dtype=dtype)
        d, fp4_indices, qs_packed = self._create_fp4_test_data(n, k, block_size, dtype)
        expected = _reference_batched_block_scaled_mmt_fp4(a, d, qs_packed, block_size)

        # Kernel expects unpacked FP4 indices
        actual = batched_block_scaled_mmt_fp4(a, d, fp4_indices)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_with_planar_quantized_tensor(self):
        """Test integration with PlanarQuantizedTensor and BlockScaledFp4Layout."""
        torch.manual_seed(42)

        batch_size, m, n, k = 2, 32, 128, 256
        block_size = 32
        dtype = torch.float32

        a = torch.randn(batch_size, m, k, dtype=dtype)
        d, fp4_indices, qs_packed = self._create_fp4_test_data(n, k, block_size, dtype)
        quantized_tensor = self._create_planar_quantized_tensor(n, k, d, qs_packed, block_size)

        # Reference expects packed data
        expected = _reference_batched_block_scaled_mmt_fp4(
            a, quantized_tensor.layout.d, quantized_tensor.layout.qs_bit_packed, block_size
        )

        actual = batched_block_scaled_mmt_fp4(
            a, quantized_tensor.layout.d, quantized_tensor.layout.qs
        )

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_power_of_two_scales(self):
        """Test with power-of-two scales (integer exponents)."""
        torch.manual_seed(42)

        batch_size, m, n, k = 1, 16, 64, 128
        block_size = 32
        dtype = torch.float32

        a = torch.randn(batch_size, m, k, dtype=dtype)
        d_exponents, fp4_indices, qs_packed = self._create_fp4_test_data(
            n, k, block_size, dtype, use_power_of_two_scale=True
        )

        # For the reference, convert exponents to actual scales
        d_actual = torch.pow(2.0, d_exponents)
        expected = _reference_batched_block_scaled_mmt_fp4(
            a, d_actual, qs_packed, block_size
        )

        # Kernel expects unpacked indices and float scales
        d_float = convert_fp4_scales_to_float(d_exponents, use_power_of_two_scale=True)
        actual = batched_block_scaled_mmt_fp4(a, d_float, fp4_indices)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
