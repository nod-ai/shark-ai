# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.types import *
from sharktank.types.layout_utils import (
    saturate_cast,
    pack_fp4_e2m1_to_uint8,
    unpack_uint8_to_fp4_e2m1,
)
from sharktank.types.ocp_floats import (
    float32_to_fp4_e2m1,
    fp4_e2m1_to_float32,
)
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
from sharktank.utils.testing import TempDirTestBase


class StaticScaledQuantizerTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _roundtrip(self, it, suffix=""):
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

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
        layout = qt_value.unpack()
        dequant_value = layout.dequant()
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
        layout = qt_value.unpack()
        dequant_value = layout.dequant()
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
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
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
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)


class DynamicScaledQuantizerTest(TempDirTestBase):
    def _roundtrip(self, it, suffix=""):
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

    def testQuantDequantInt(self):
        qr = DynamicScaledQuantizer(dtype=torch.int8)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf16(self):
        qr = DynamicScaledQuantizer(dtype=torch.float16)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf8fn(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fn)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuantDequantf8fnuz(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fnuz)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
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


class StaticScaledFP4QuantizerTest(TempDirTestBase):
    def _roundtrip(self, it, suffix=""):
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

    def testFP4QuantDequant(self):
        scale = torch.tensor(0.5, dtype=torch.float32)
        ssq = StaticScaledQuantizer(scale=scale, dtype=torch.float32)
        ssq = self._roundtrip(ssq, "_fp4_ssq")

        # Original values that will be scaled before FP4 quantization
        orig_value = torch.tensor([2.0, 4.0, 6.0, 8.0, 1.0, 3.0, -2.0, -4.0])

        # Manually implement FP4 quantization with scaling
        # First scale the values
        scaled_values = orig_value * scale  # [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, -1.0, -2.0]
        fp4_indices = float32_to_fp4_e2m1(scaled_values)
        packed_fp4 = pack_fp4_e2m1_to_uint8(fp4_indices)

        # Create a quantized tensor with packed FP4 data
        qt_value = PlanarQuantizedTensor(
            shape=list(orig_value.shape),
            name="test_fp4",
            layout=TensorScaledLayout(
                shape=list(orig_value.shape),
                d=ssq.reciprocal_scale,  # 1/0.5 = 2.0
                qs=packed_fp4,
                dtype=orig_value.dtype,
            ),
        )

        # Test roundtrip
        qt_value = self._roundtrip(qt_value, "_fp4_qt_value")

        # Manually implement FP4 dequantization with scaling
        layout = qt_value.unpack()
        packed_data = layout.qs
        unpacked_indices = unpack_uint8_to_fp4_e2m1(packed_data)
        fp4_values = fp4_e2m1_to_float32(unpacked_indices)
        dequant_value = (
            fp4_values * layout.d
        )  # Apply reciprocal scale to get back original

        torch.testing.assert_close(orig_value, dequant_value, atol=0.0, rtol=0.0)

    def testFP4QuantDequantApproximation(self):

        scale = torch.tensor(0.25, dtype=torch.float32)
        ssq = StaticScaledQuantizer(scale=scale, dtype=torch.uint8)
        ssq = self._roundtrip(ssq, "_fp4_approx_ssq")

        # Values that will require approximation after scaling
        orig_value = torch.tensor([2.5, 5.0, 7.5, 10.0, 1.25, 3.75, -2.5, -5.0])

        expected_dequant = torch.tensor([2.0, 4.0, 8.0, 8.0, 2.0, 4.0, -2.0, -4.0])

        scaled_values = orig_value * scale
        fp4_indices = float32_to_fp4_e2m1(scaled_values)
        packed_fp4 = pack_fp4_e2m1_to_uint8(fp4_indices)

        qt_value = PlanarQuantizedTensor(
            shape=list(orig_value.shape),
            name="test_fp4_approx",
            layout=TensorScaledLayout(
                shape=list(orig_value.shape),
                d=ssq.reciprocal_scale,
                qs=packed_fp4,
                dtype=orig_value.dtype,
            ),
        )

        qt_value = self._roundtrip(qt_value, "_fp4_approx_qt_value")

        layout = qt_value.unpack()
        packed_data = layout.qs
        unpacked_indices = unpack_uint8_to_fp4_e2m1(packed_data)
        fp4_values = fp4_e2m1_to_float32(unpacked_indices)
        dequant_value = fp4_values * layout.d

        torch.testing.assert_close(expected_dequant, dequant_value, atol=0.0, rtol=0.0)

    def testFP4BlockQuantization(self):
        orig_value = torch.randn(128) * 3.0

        # Block quantize with power-of-two scales
        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_power_of_two_scale=True, name="fp4_quantizer"
        )
        quantized_tensor = quantizer.quantize(orig_value, name="fp4_quantized")

        self.assertIsInstance(quantized_tensor, PlanarQuantizedTensor)
        layout = quantized_tensor.unpack()
        self.assertIsInstance(layout, BlockScaledFp4Layout)
        self.assertEqual(len(layout.d), 4)
        self.assertTrue(layout.d.dtype == torch.int32)

        # Dequantize
        dequantized = quantized_tensor.unpack().dequant()

        self.assertEqual(dequantized.shape, orig_value.shape)

        # Test with different block size
        quantizer_16 = DynamicFp4BlockQuantizer(
            block_size=16,
            use_power_of_two_scale=True,
            name="fp4_quantizer",
        )
        quantized_tensor_16 = quantizer_16.quantize(orig_value, name="fp4_quantized")

        layout_16 = quantized_tensor_16.unpack()
        self.assertEqual(len(layout_16.d), 8)


if __name__ == "__main__":
    unittest.main()
