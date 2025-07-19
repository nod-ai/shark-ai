# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from sharktank.types import DynamicFp4BlockQuantizer, Slice
from sharktank.utils.testing import assert_tensor_close
from sharktank import ops


class TestExtractSlice_BlockScaledFp4Layout:
    n_values_per_byte = 2
    block_size = n_values_per_byte * 3

    @pytest.mark.parametrize(
        "slice_",
        [
            (None,),
            tuple(),
            (slice(None), slice(None), slice(None)),
            (slice(None, -1), slice(None, -1), slice(None, -block_size)),
            (slice(-2, -1), slice(-3, -2), slice(-2 * block_size, -block_size)),
            (slice(1, 1)),
            (
                None,
                slice(None),
                slice(2),
                None,
                slice(block_size, 3 * block_size),
            ),
        ],
    )
    def test_extract_slice_block_scaled_f4_quantization(
        self, deterministic_random_seed, slice_: Slice
    ):
        dequantized_dtype = torch.float32
        quantizer = DynamicFp4BlockQuantizer(
            dtype=torch.float32,
            block_size=self.block_size,
            use_fe8m0_scale=True,
        )
        dequantized_input = torch.rand(
            [3, 5, quantizer.block_size * 7], dtype=dequantized_dtype
        )
        quantized_input = quantizer.quantize(dequantized_input)

        # Roundtrip the dequantization and quantization to make sure that
        # the tensor gets quantized without any loss of precision.
        dequantized_input = quantized_input.to_planar().layout.dequant(
            dtype=dequantized_dtype
        )
        quantized_input_roundtripped = quantizer.quantize(dequantized_input)
        # Make sure roundtripping is sane.
        assert ops.equal(quantized_input, quantized_input_roundtripped)

        expected_result = dequantized_input[slice_]
        actual_result = quantized_input[slice_]
        assert_tensor_close(actual_result, expected_result, rtol=0, atol=0)
