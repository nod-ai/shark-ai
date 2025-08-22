# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import torch

from sharktank.layers import LinearLayer
from sharktank.types import Theta, StaticScaledQuantizer
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.layers.quarot import QuaRotTransform, apply_hadamard_transform

logger = logging.getLogger(__name__)


class QuaRotTransformTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def test_hadamard_transform_invalid_size(self):
        """Test that non-power-of-2 dimensions raise ValueError."""
        x = torch.randn(2, 3, 5)
        with self.assertRaises(ValueError):
            apply_hadamard_transform(x)

    def test_quarot_transform_forward_inverse(self):
        """Test that QuaRotTransform forward and inverse are correct."""
        hidden_dim = 16
        x = torch.randn(3, 4, hidden_dim)

        signs = torch.randint(0, 2, (hidden_dim,), dtype=torch.int8) * 2 - 1
        theta = Theta(
            [
                DefaultPrimitiveTensor(name="quarot_signs", data=signs),
            ]
        )
        transform = QuaRotTransform(theta, hidden_dim, seed=456)

        x_rotated = transform.forward(x)
        x_recovered = transform.inverse(x_rotated)

        torch.testing.assert_close(x_recovered, x, atol=1e-4, rtol=1e-4)

    def test_quarot_transform_golden(self):
        """Test QuaRotTransform with deterministic values."""
        hidden_dim = 32
        x = torch.zeros(1, 1, 32, dtype=torch.float32)
        x[0, 0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        signs = torch.ones(32, dtype=torch.int8)
        signs[:4] = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
        theta = Theta(
            [
                DefaultPrimitiveTensor(name="quarot_signs", data=signs),
            ]
        )
        transform = QuaRotTransform(theta, hidden_dim, seed=42)

        x_rotated = transform.forward(x)

        sqrt_32 = 32**0.5
        expected_first_few = torch.tensor(
            [-2.0 / sqrt_32, 10.0 / sqrt_32, 0.0 / sqrt_32, -4.0 / sqrt_32],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            x_rotated[0, 0, :4], expected_first_few, atol=1e-4, rtol=1e-4
        )


class QuaRotLinearLayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def test_quarot_linear_layer_with_quantization(self):
        """Test QuaRot LinearLayer with existing quantization."""
        batch_size, seq_len, hidden_dim = 2, 4, 32
        out_features = 64

        x = torch.randn(batch_size, seq_len, hidden_dim)

        weight = torch.randn(out_features, hidden_dim)
        bias = torch.randn(out_features)

        input_scale = torch.tensor(0.5, dtype=torch.float32)
        input_quantizer = StaticScaledQuantizer(
            name="q_input", scale=input_scale, dtype=torch.int8
        )

        theta = Theta(
            [
                input_quantizer,
                DefaultPrimitiveTensor(name="weight", data=weight),
                DefaultPrimitiveTensor(name="bias", data=bias),
            ]
        )

        quarot_layer = LinearLayer(theta, use_quarot=True, fake_quant=True)
        quarot_output = quarot_layer(x)

        self.assertEqual(quarot_output.shape, (batch_size, seq_len, out_features))

        self.assertFalse(torch.isnan(quarot_output).any())
        self.assertFalse(torch.isinf(quarot_output).any())

    def test_quarot_linear_layer_golden(self):
        """Test QuaRot LinearLayer with deterministic values."""
        hidden_dim = 64

        # Create deterministic input with padding - use quantization-friendly values
        x = torch.zeros(1, 1, 64, dtype=torch.float32)
        x[0, 0, :4] = torch.tensor([8.0, 0.0, -8.0, 0.0], dtype=torch.float32)

        theta_temp = Theta([])
        transform = QuaRotTransform(theta_temp, hidden_dim, seed=42)

        identity = (
            torch.eye(64, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, 64, 64]
        weight = transform.inverse(identity).squeeze()
        bias = torch.zeros(64, dtype=torch.float32)
        bias[:4] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

        theta = Theta(
            [
                DefaultPrimitiveTensor(name="weight", data=weight),
                DefaultPrimitiveTensor(name="bias", data=bias),
            ]
        )

        quarot_layer = LinearLayer(theta, use_quarot=True)
        quarot_output = quarot_layer(x)

        expected = torch.tensor(
            [[[8.0 + 0.1, 0.0 + 0.2, -8.0 + 0.3, 0.0 + 0.4]]], dtype=torch.float32
        )

        torch.testing.assert_close(
            quarot_output[:, :, :4], expected, atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
