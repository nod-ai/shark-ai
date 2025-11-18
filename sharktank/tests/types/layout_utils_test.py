# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from amdsharktank.types.quantizers import DynamicFp4BlockQuantizer
from amdsharktank.types.layout_utils import (
    debug_map_tensor_as_hex_string,
    interleave_linear_i4_block,
    linearize_interleaved_i4_block,
    pack_fp4_e2m1_to_uint8,
    promote_linear_i2_block_to_i8,
    promote_linear_i4_block_to_i8,
    promote_linear_i6_block_to_i8,
    unpack_uint8_to_fp4_e2m1,
)
from amdsharktank.types.layouts import BlockScaledFp4Layout
from amdsharktank.types.ocp_floats import (
    FloatingPointFormat,
    get_fp4_lookup_table,
    _FP4_E2M1_TO_FP32,
    float32_to_fp4_e2m1,
    fp4_e2m1_to_float32,
)
from amdsharktank.types.tensors import PlanarQuantizedTensor


class FP4Tests(unittest.TestCase):
    def test_fp4_hardcoded_vs_generated_lookup_table(self):
        """Test that hardcoded lookup table matches generated one."""

        generated_table = get_fp4_lookup_table(FloatingPointFormat.E2M1)

        # Compare with hardcoded table
        torch.testing.assert_close(
            _FP4_E2M1_TO_FP32,
            generated_table,
            atol=0.0,
            rtol=0.0,
            msg="Hardcoded FP4 E2M1 lookup table doesn't match generated version",
        )

    def test_fp4_conversion_roundtrip(self):
        """Test that float32 -> FP4 -> float32 conversion works for representable values."""
        # Test values that should be exactly representable in FP4 E2M1
        test_values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        )

        # Convert to FP4 indices and back
        fp4_indices = float32_to_fp4_e2m1(test_values)
        recovered_values = fp4_e2m1_to_float32(fp4_indices)

        torch.testing.assert_close(test_values, recovered_values, atol=0.0, rtol=0.0)

    def test_fp4_conversion_approximation(self):
        """Test that float32 -> FP4 conversion finds nearest representable values."""
        # Test values that need to be approximated
        test_values = torch.tensor([0.25, 0.75, 1.25, 2.5, 5.0], dtype=torch.float32)
        expected_approximations = torch.tensor(
            [0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32
        )

        fp4_indices = float32_to_fp4_e2m1(test_values)
        approximated_values = fp4_e2m1_to_float32(fp4_indices)

        torch.testing.assert_close(
            approximated_values, expected_approximations, atol=0.0, rtol=0.0
        )

    def test_fp4_packing_unpacking(self):
        """Test that FP4 packing and unpacking works correctly."""
        fp4_values = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8
        )
        packed = pack_fp4_e2m1_to_uint8(fp4_values)
        self.assertEqual(packed.shape[-1], fp4_values.shape[-1] // 2)
        unpacked = unpack_uint8_to_fp4_e2m1(packed)

        torch.testing.assert_close(unpacked, fp4_values, atol=0, rtol=0)

    def test_fp4_packing_bit_layout(self):
        """Test that FP4 packing follows expected bit layout."""
        fp4_values = torch.tensor(
            [0b0001, 0b0010, 0b1111, 0b1000], dtype=torch.uint8
        )  # 1, 2, 15, 8

        packed = pack_fp4_e2m1_to_uint8(fp4_values)

        # Expected: [0b00100001, 0b10001111] = [0x21, 0x8F]
        # First byte: high_nibble=0010, low_nibble=0001 -> 0x21
        # Second byte: high_nibble=1000, low_nibble=1111 -> 0x8F
        expected_packed = torch.tensor([0x21, 0x8F], dtype=torch.uint8)

        torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)

    def test_fp4_edge_cases(self):
        """Test FP4 conversion with edge cases."""
        # Test clamping of out-of-range values
        large_values = torch.tensor([100.0, -100.0, 1000.0], dtype=torch.float32)
        fp4_indices = float32_to_fp4_e2m1(large_values)
        clamped_values = fp4_e2m1_to_float32(fp4_indices)

        # Should clamp to representable values
        expected_max = torch.tensor([6.0, -6.0, 6.0], dtype=torch.float32)
        torch.testing.assert_close(clamped_values, expected_max, atol=0.0, rtol=0.0)

    def test_fp4_batch_processing(self):
        """Test FP4 operations work with batched tensors."""
        batch_values = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, -1.0]], dtype=torch.float32
        )

        fp4_indices = float32_to_fp4_e2m1(batch_values)
        recovered_values = fp4_e2m1_to_float32(fp4_indices)

        expected = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 4.0, 6.0, -1.0]], dtype=torch.float32
        )

        torch.testing.assert_close(recovered_values, expected, atol=0.0, rtol=0.0)


class I4Shuffle(unittest.TestCase):
    def test_linearize_interleaved_i4_block(self):
        # Linearize.
        input_data = torch.tensor(
            [0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7], dtype=torch.uint8
        ).unsqueeze(0)
        linear = linearize_interleaved_i4_block(input_data)
        self.assertEqual(
            r"[['0x10', '0x32', '0x54', '0x76', '0x98', '0xba', '0xdc', '0xfe']]",
            repr(debug_map_tensor_as_hex_string(linear)),
        )

        # Invert back to interleaved.
        interleaved = interleave_linear_i4_block(linear)
        self.assertEqual(
            r"[['0x80', '0x91', '0xa2', '0xb3', '0xc4', '0xd5', '0xe6', '0xf7']]",
            repr(debug_map_tensor_as_hex_string(interleaved)),
        )

    def test_promote_i4_block_to_i8_unsigned(self):
        # Start with linear data.
        linear_i4_data = torch.tensor(
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8
        ).unsqueeze(0)
        r0 = promote_linear_i4_block_to_i8(linear_i4_data)
        self.assertEqual(r0.dtype, torch.uint8)
        torch.testing.assert_close(
            torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
                dtype=torch.uint8,
            ),
            r0,
        )

    def test_promote_i4_block_to_i8_signed(self):
        # Start with linear data.
        linear_i4_data = (
            torch.tensor(
                [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8
            )
            .unsqueeze(0)
            .view(torch.uint8)
        )
        r0 = promote_linear_i4_block_to_i8(linear_i4_data, signed=True)
        self.assertEqual(r0.dtype, torch.int8)
        torch.testing.assert_close(
            torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]],
                dtype=torch.int8,
            ),
            r0,
        )

    def test_promote_i2_block_to_i8(self):
        data = torch.tensor([[0xC1, 0xB2, 0xA3, 0x94, 0x85]], dtype=torch.uint8)
        expected = torch.tensor(
            # fmt: off
            [[
                1, 0, 0, 3,  # 0xC1
                2, 0, 3, 2,  # 0xB2
                3, 0, 2, 2,  # 0xA3
                0, 1, 1, 2,  # 0x94
                1, 1, 0, 2   # 0x85
            ]],
            dtype=torch.uint8,
            # fmt: on
        )
        r0 = promote_linear_i2_block_to_i8(data)
        torch.testing.assert_close(r0, expected)

    def test_promote_i6_block_to_i8(self):
        # High 2 bit values: 0, 3, 1, 3, 1, 3, 0, 3
        high = torch.tensor([[0xDC, 0xCD]], dtype=torch.uint8)
        # Low 4 bit values:
        # '0xb', '0xc', '0x2', '0x3', '0x1', '0x1', '0x6', '0x7'
        low = torch.tensor([[0xCB, 0x32, 0x11, 0x76]], dtype=torch.uint8)
        r0 = promote_linear_i6_block_to_i8(high, low)
        r_debug = repr(debug_map_tensor_as_hex_string(r0))
        self.assertEqual(
            r_debug, "[['0xb', '0x3c', '0x12', '0x33', '0x11', '0x31', '0x6', '0x37']]"
        )


if __name__ == "__main__":
    unittest.main()
