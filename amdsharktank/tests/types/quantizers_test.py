# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from amdsharktank import ops
from amdsharktank.types import *
from amdsharktank.types.ocp_floats import fp4_e2m1_to_float32
from amdsharktank.types.quantizers import (
    DynamicFp4BlockQuantizer,
    StaticFp4BlockQuantizer,
)
from amdsharktank.utils.testing import TempDirTestBase


class QuantizerTestBase(TempDirTestBase):
    """Base class for quantizer tests with common utilities."""

    def _roundtrip(self, it, suffix=""):
        """Save and reload a tensor through IRPA format for serialization testing."""
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)


class StaticScaledQuantizerTest(QuantizerTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def testPerTensorRoundtrip(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            scale=torch.tensor(0.2, dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        self.assertIs(ssq.axis, None)
        self.assertEqual(ssq.scale, 0.2)
        self.assertEqual(ssq.reciprocal_scale, 5.0)
        self.assertIs(ssq.dtype, torch.float16)

    def testPerTensorQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(2.0, dtype=torch.float32), dtype=torch.uint8
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        qt_value = ssq.quantize(orig_value)
        qt_value = self._roundtrip(qt_value, "_qt_value")
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerTensorOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(2.0, dtype=torch.float32),
            offset=torch.tensor(8, dtype=torch.int8),
            dtype=torch.int8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float16)
        qt_value = ssq.quantize(orig_value)
        qt_value = self._roundtrip(qt_value, "_qt_value")
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerAxisRoundtrip(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            scale=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        self.assertEqual(ssq.axis, 1)
        torch.testing.assert_close(
            ssq.scale, torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32)
        )
        torch.testing.assert_close(
            ssq.reciprocal_scale, torch.tensor([5.0, 2.5, 1.25], dtype=torch.float32)
        )
        self.assertIs(ssq.dtype, torch.float16)

    def testPerAxisQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            # Note that the range of the third channel requires a smaller scale
            # to pass the test (otherwise, will saturate at ~30 for scale >= 4
            # or so).
            scale=torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32),
            dtype=torch.int8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor(
            [[1.0, -2.0, 3.0], [10.0, -20.0, 60.0]], dtype=torch.float16
        )
        qt_value = ssq.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)

    def testPerAxisOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            # Carefully chosen scale and offset channels that are big enough
            # to handle outliers below.
            scale=torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32),
            offset=torch.tensor([16, 127, 136], dtype=torch.uint8),
            dtype=torch.uint8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor(
            [[9.0, -11.0, 13.0], [18.0, -29.0, 40.0]], dtype=torch.float16
        )
        qt_value = ssq.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)


class DynamicScaledQuantizerTest(QuantizerTestBase):
    def testQuantDequantInt(self):
        qr = DynamicScaledQuantizer(dtype=torch.int8)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf16(self):
        qr = DynamicScaledQuantizer(dtype=torch.float16)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf8fn(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fn)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuantDequantf8fnuz(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fnuz)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        dequant_value = qt_value.unpack().dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuarkF8Hell(self):
        # we use hardcoded values here because they're representative of actual values from a quark model
        scale = torch.tensor(0.0118, dtype=torch.float64)
        orig = torch.tensor(
            [
                -58,
                -48,
                -70,
                53,
                -53,
                76,
                -71,
                -90,
                50,
                77,
                62,
                -98,
                66,
                -54,
                55,
                -80,
                -66,
                -62,
                -61,
                -56,
                56,
                -67,
                79,
                -60,
                -71,
                42,
                72,
                -73,
                91,
                63,
                124,
                -128,
            ],
            dtype=torch.int8,
        )
        # mirrors dequant logic  in quark and our importer
        orig = orig.view(torch.float8_e4m3fn)
        orig = (orig.to(torch.float64) * scale).to(torch.float16)
        # Note that for fnuz  we have to do scale*2 to account for the difference between types
        # We specify the reciprocal scale explicitly to avoid adding more floating point error noise
        fnuz = StaticScaledQuantizer(
            name="dopoo",
            scale=1.0 / (scale * 2),
            reciprocal_scale=scale * 2,
            offset=None,
            dtype=torch.float8_e4m3fnuz,
        )
        fn = StaticScaledQuantizer(
            name="poodoo",
            scale=1.0 / scale,
            reciprocal_scale=scale,
            offset=None,
            dtype=torch.float8_e4m3fn,
        )
        fnuz_quant = fnuz.quantize(orig)
        fn_quant = fn.quantize(orig)

        dequant_fnuz = fnuz_quant.unpack().dequant()
        dequant_fn = fn_quant.unpack().dequant()

        # redundant asserts for sanity
        torch.testing.assert_close(
            orig.to(torch.float16), dequant_fnuz, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            orig.to(torch.float16), dequant_fn, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(dequant_fnuz, dequant_fn, atol=1e-3, rtol=1e-3)


class Fp4BlockQuantizerTestBase(QuantizerTestBase):
    """Base class for FP4 quantizer tests with common test data and utilities."""

    @staticmethod
    def get_fp4_exact_values():
        """Get test values that are exactly representable in FP4."""
        return torch.tensor(
            [2.0, 4.0, 6.0, 6.0, 1.0, 3.0, -2.0, -4.0], dtype=torch.float32
        )

    @staticmethod
    def get_fp4_approximate_values():
        """Get test values that require approximation in FP4."""
        return torch.tensor(
            [2.5, 5.0, 7.5, 10.0, 1.25, 3.75, -2.5, -5.0], dtype=torch.float32
        )

    def assert_layout_properties(self, layout, expected_block_count, scale_dtype):
        """Assert common properties of FP4 block scaled layouts."""
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        self.assertEqual(layout.d.numel(), expected_block_count)
        self.assertEqual(layout.d.dtype, scale_dtype)


class DynamicFP4BlockQuantizerTest(Fp4BlockQuantizerTestBase):
    def testFP4QuantizerKernel(self):
        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_fe8m0_scale=True, name="fp4_quantizer"
        )
        quantizer = self._roundtrip(quantizer, "_fp4_quantizer")

        # Use 32 values for block size 32 test
        orig_value = torch.tensor(
            [
                2.0,
                4.0,
                6.0,
                6.0,
                1.0,
                3.0,
                -2.0,
                -4.0,
                1.5,
                2.0,
                3.0,
                4.0,
                0.5,
                1.0,
                -1.5,
                -3.0,
                4.0,
                6.0,
                2.0,
                3.0,
                1.5,
                4.0,
                -2.0,
                -6.0,
                3.0,
                1.0,
                2.0,
                4.0,
                6.0,
                1.5,
                -1.0,
                -4.0,
            ],
            dtype=torch.float32,
        )

        qt_value = quantizer.quantize(orig_value, name="test_fp4")
        qt_value = self._roundtrip(qt_value, "_fp4_qt_value")

        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        dequant_value = layout.dequant()

        torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)

        # Test that the quantizer works for exactly scaled by powers of two
        orig_value2 = orig_value * 64
        qt_value = quantizer.quantize(orig_value2, name="test_fp4")
        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)

        torch.testing.assert_close(
            orig_value.view(1, 32), fp4_e2m1_to_float32(layout.qs), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            torch.tensor([[127 + 6]]), layout.d, atol=0.0, rtol=0.0, check_dtype=False
        )

        orig_value3 = orig_value / (2**65.0)
        qt_value = quantizer.quantize(orig_value3, name="test_fp4")
        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)

        torch.testing.assert_close(
            orig_value.view(1, 32), fp4_e2m1_to_float32(layout.qs), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            torch.tensor([[127 - 65]]), layout.d, atol=0.0, rtol=0.0, check_dtype=False
        )

    def testFP4QuantDequant(self):
        quantizer = DynamicFp4BlockQuantizer(
            block_size=8,
            use_fe8m0_scale=True,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantizer = self._roundtrip(quantizer, "_fp4_quantizer")

        # Use 8 values for block size 8 test
        orig_value = self.get_fp4_exact_values()

        qt_value = quantizer.quantize(orig_value, name="test_fp4")
        qt_value = self._roundtrip(qt_value, "_fp4_qt_value")

        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        dequant_value = layout.dequant()

        torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)

    def testFP4QuantDequantApproximation(self):
        quantizer = DynamicFp4BlockQuantizer(
            block_size=8,
            use_fe8m0_scale=False,
            name="fp4_approx_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantizer = self._roundtrip(quantizer, "_fp4_approx_quantizer")

        orig_values = self.get_fp4_approximate_values()

        qt_value = quantizer.quantize(orig_values, name="test_fp4_approx")
        qt_value = self._roundtrip(qt_value, "_fp4_approx_qt_value")
        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        dequant_value = layout.dequant()

        # The error will be quite large because of the imprecision of fp4
        torch.testing.assert_close(orig_values, dequant_value, atol=1.0, rtol=1.0)

    def testFP4BlockQuantization(self):
        orig_value = torch.randn(128, dtype=torch.float32) * 3.0

        quantizer = DynamicFp4BlockQuantizer(
            block_size=32,
            use_fe8m0_scale=True,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantized_tensor = quantizer.quantize(orig_value, name="fp4_quantized")

        self.assertIsInstance(quantized_tensor, PlanarQuantizedTensor)
        layout = quantized_tensor.unpack()
        self.assert_layout_properties(
            layout, expected_block_count=4, scale_dtype=torch.uint8
        )

        # Dequantize
        dequantized = quantized_tensor.unpack().dequant()

        self.assertEqual(dequantized.shape, orig_value.shape)

        # Test with different block size
        quantizer_16 = DynamicFp4BlockQuantizer(
            block_size=16,
            use_fe8m0_scale=True,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantized_tensor_16 = quantizer_16.quantize(orig_value, name="fp4_quantized")

        layout_16 = quantized_tensor_16.unpack()
        self.assertEqual(len(layout_16.d), 8)

    def testFp4BlockQuantization2(self):
        """Test FP4 block quantization with configurable block size and FE8M0 scales."""
        original_data = torch.randn(64, dtype=torch.float32) * 4.0

        # FE8M0 scales
        quantizer = DynamicFp4BlockQuantizer(
            block_size=32,
            use_fe8m0_scale=True,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantized_tensor = quantizer.quantize(original_data, name="fp4_quantized")

        self.assertIsInstance(quantized_tensor, PlanarQuantizedTensor)
        layout = quantized_tensor.unpack()
        self.assert_layout_properties(
            layout, expected_block_count=2, scale_dtype=torch.uint8
        )

        dequantized = quantized_tensor.unpack().dequant()

        self.assertEqual(dequantized.shape, original_data.shape)

        # Float scales
        quantizer_float = DynamicFp4BlockQuantizer(
            block_size=32,
            use_fe8m0_scale=False,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantized_tensor_float = quantizer_float.quantize(
            original_data, name="fp4_quantized"
        )

        layout_float = quantized_tensor_float.unpack()
        self.assertTrue(layout_float.d.dtype == torch.float32)

        dequantized_float = quantized_tensor_float.unpack().dequant()

        self.assertEqual(dequantized_float.shape, original_data.shape)

    def testFp4ConfigurableBlockSize(self):
        """Test FP4 block quantization with different block sizes."""
        original_data = torch.randn(60, dtype=torch.float32) * 4.0

        quantizer = DynamicFp4BlockQuantizer(
            block_size=6,
            use_fe8m0_scale=True,
            name="fp4_quantizer",
            use_amdsharktank_kernel=False,
        )
        quantized_tensor = quantizer.quantize(original_data, name="fp4_quantized")

        self.assertIsInstance(quantized_tensor, PlanarQuantizedTensor)
        layout = quantized_tensor.unpack()
        self.assert_layout_properties(
            layout, expected_block_count=10, scale_dtype=torch.uint8
        )

        dequantized = quantized_tensor.unpack().dequant()

        self.assertEqual(dequantized.shape, original_data.shape)


class StaticFp4BlockQuantizerTest(Fp4BlockQuantizerTestBase):
    def testStaticFp4QuantDequant(self):
        orig_value = self.get_fp4_exact_values()
        scales = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1)
        static_quantizer = StaticFp4BlockQuantizer(
            scales=scales,
            block_size=8,
            use_fe8m0_scale=False,
            name="static_fp4_quantizer",
        )
        static_quantizer = self._roundtrip(static_quantizer, "_static_fp4_quantizer")

        # Quantize with static quantizer
        qt_value = static_quantizer.quantize(orig_value, name="test_static_fp4")
        qt_value = self._roundtrip(qt_value, "_static_fp4_qt_value")

        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        dequant_value = layout.dequant()

        # Should match original values exactly for representable FP4 values
        torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)

    def testStaticFp4PowerOfTwoScales(self):
        orig_value = torch.randn(64, dtype=torch.float32) * 4.0

        # Create static quantizer with manually set FE8M0 scales
        # For 64 elements with block_size=32, we need 2 blocks
        scales = torch.tensor([130, 131], dtype=torch.uint8).unsqueeze(-1)

        static_quantizer = StaticFp4BlockQuantizer(
            scales=scales,
            block_size=32,
            use_fe8m0_scale=True,
            name="static_fp4_p2_quantizer",
        )
        static_quantizer = self._roundtrip(static_quantizer, "_static_fp4_p2_quantizer")
        qt_value = static_quantizer.quantize(orig_value, name="test_static_fp4_p2")
        qt_value = self._roundtrip(qt_value, "_static_fp4_p2_qt_value")

        layout = qt_value.unpack()
        self.assert_layout_properties(
            layout, expected_block_count=2, scale_dtype=torch.uint8
        )

        dequant_value = layout.dequant()
        self.assertEqual(dequant_value.shape, orig_value.shape)

    def testStaticFp4TwoDimensionalScales(self):
        orig_value = self.get_fp4_exact_values().reshape(2, 4)
        scales = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32).unsqueeze(
            -1
        )
        static_quantizer = StaticFp4BlockQuantizer(
            scales=scales,
            block_size=2,
            use_fe8m0_scale=False,
            name="static_fp4_quantizer",
        )
        static_quantizer = self._roundtrip(static_quantizer, "_static_fp4_p2_quantizer")
        qt_value = static_quantizer.quantize(orig_value, name="test_static_fp4_p2")
        qt_value = self._roundtrip(qt_value, "_static_fp4_p2_qt_value")

        layout = qt_value.unpack()
        self.assert_layout_properties(
            layout, expected_block_count=4, scale_dtype=torch.float32
        )

        dequant_value = layout.dequant()
        self.assertEqual(dequant_value.shape, orig_value.shape)

    def testStaticFp4DifferentBlockSizes(self):
        for block_size in [6, 10, 12, 30]:
            orig_value = torch.randn(60, dtype=torch.float32) * 4.0

            # Create appropriate scales for this block size
            expected_num_blocks = (60 + block_size - 1) // block_size
            # Use simple fe8m0
            scales = torch.randint(
                125, 130, (expected_num_blocks,), dtype=torch.uint8
            ).unsqueeze(-1)

            static_quantizer = StaticFp4BlockQuantizer(
                scales=scales,
                block_size=block_size,
                use_fe8m0_scale=True,
                name=f"static_fp4_bs{block_size}",
            )
            qt_value = static_quantizer.quantize(
                orig_value, name=f"test_static_fp4_bs{block_size}"
            )

            layout = qt_value.unpack()
            self.assertIsInstance(layout, BlockScaledFp4Layout)
            self.assertEqual(len(layout.d), expected_num_blocks)

            # Verify scales are preserved
            torch.testing.assert_close(layout.d, scales)

            # Test basic functionality
            dequant_value = layout.dequant()
            self.assertEqual(dequant_value.shape, orig_value.shape)

            # Test that serialization works
            static_quantizer_rt = self._roundtrip(
                static_quantizer, f"_static_fp4_bs{block_size}"
            )
            self.assertEqual(static_quantizer_rt.block_size, block_size)
            torch.testing.assert_close(static_quantizer_rt.scales, scales)

    def testReplicatedStaticFp4Quantizer(self):
        """Test that replicated static FP4 quantizer works correctly."""
        orig_value = self.get_fp4_exact_values()
        scales = torch.tensor([1.0], dtype=torch.float32).unsqueeze(-1)

        # Create a static quantizer with replicated scales
        static_quantizer = StaticFp4BlockQuantizer(
            scales=scales,
            block_size=8,
            use_fe8m0_scale=False,
            name="replicated_static_fp4_quantizer",
        )
        static_quantizer = self._roundtrip(
            static_quantizer, "_replicated_static_fp4_quantizer"
        )

        # Quantize with static quantizer
        qt_value = static_quantizer.quantize(
            orig_value, name="test_replicated_static_fp4"
        )
        # TODO: Enable after generalizating add_to_archive
        # qt_value = self._roundtrip(qt_value, "_replicated_static_fp4_qt_value")

        layout = qt_value.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        dequant_value = layout.dequant()

        # Should match original values exactly for representable FP4 values
        torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)

        replicated_quantizer = ReplicatedTensor(ts=static_quantizer, shard_count=4)
        orig_values_replicated = ReplicatedTensor(ts=orig_value, shard_count=4)
        qt_value_replicated = ops.quantize(
            orig_values_replicated,
            replicated_quantizer,
            name="test_replicated_static_fp4",
        )
        # TODO: Enable after generalizating add_to_archive
        # qt_value_replicated = self._roundtrip(qt_value_replicated, "_replicated_static_fp4_qt_value")

        for qt_shard in qt_value_replicated.shards:
            layout = qt_shard.unpack()
            self.assertIsInstance(layout, BlockScaledFp4Layout)
            dequant_value = layout.dequant()
            torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
