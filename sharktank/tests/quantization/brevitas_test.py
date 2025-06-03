# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import torch
from torch import nn
from parameterized import parameterized

from sharktank.layers import LinearLayer, QuantizationLayer
from sharktank.types import Theta, DefaultPrimitiveTensor, StaticScaledQuantizer
from sharktank.utils.testing import make_rand_torch
from sharktank.quantization.brevitas import quantize

try:
    import brevitas
    BREVITAS_AVAILABLE = True
except ImportError:
    BREVITAS_AVAILABLE = False


@unittest.skipUnless(BREVITAS_AVAILABLE, "Brevitas not available")
class BrevitasQuantizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def _create_linear_layer(self, input_size, output_size, with_bias=True):
        """Helper to create a SHARK LinearLayer with random weights."""
        weight = make_rand_torch([output_size, input_size], dtype=torch.float32)
        nn.init.uniform_(weight, -0.5, 0.5)
        
        tensors = [DefaultPrimitiveTensor(data=weight, name="weight")]
        
        if with_bias:
            bias = make_rand_torch([output_size], dtype=torch.float32)
            nn.init.uniform_(bias, -0.1, 0.1)
            tensors.append(DefaultPrimitiveTensor(data=bias, name="bias"))
        
        return LinearLayer(Theta(tensors))

    def test_linear_layer_quantization_functional(self):
        """Test SHARK LinearLayer vs PyTorch and quantized SHARK vs unquantized SHARK."""
        input_size, output_size, batch_size = 128, 64, 8
        
        weight = make_rand_torch([output_size, input_size], dtype=torch.float32)
        bias = make_rand_torch([output_size], dtype=torch.float32)
        
        # Create SHARK LinearLayer
        weight_tensor = DefaultPrimitiveTensor(data=weight, name="weight")
        bias_tensor = DefaultPrimitiveTensor(data=bias, name="bias")
        shark_linear = LinearLayer(Theta([weight_tensor, bias_tensor]))
        
        # Create equivalent PyTorch layer
        torch_linear = nn.Linear(input_size, output_size)
        torch_linear.weight.data = weight.clone()
        torch_linear.bias.data = bias.clone()
        
        x = make_rand_torch([batch_size, input_size], dtype=torch.float32)
        
        # Test SHARK vs PyTorch
        shark_output = shark_linear(x)
        torch_output = torch_linear(x)
        torch.testing.assert_close(shark_output, torch_output, atol=1e-6, rtol=1e-6)
        
        # Test quantized SHARK vs unquantized SHARK
        # Wrap in Sequential so FX tracing treats LinearLayer as a module
        model_with_shark = nn.Sequential(shark_linear)
        quantized_model = quantize(model_with_shark)
        
        # Check if the quantized model contains brevitas quantized layers
        found_quant_layer = False
        for module in quantized_model.modules():
            if isinstance(module, brevitas.nn.QuantLinear):
                found_quant_layer = True
                break
        self.assertTrue(found_quant_layer, "Expected to find qnn.QuantLinear in quantized model")
        
        # Quantized should be reasonably close to unquantized
        with torch.no_grad():
            quantized_output = quantized_model(x)
        
        torch.testing.assert_close(shark_output, quantized_output, atol=0.1, rtol=0.05)
        

    @parameterized.expand([
        (32, 16, True),
        (64, 32, False),
        (128, 64, True),
    ])
    def test_linear_quantization_accuracy(self, input_size, output_size, with_bias):
        """Test SHARK LinearLayer vs PyTorch and quantized SHARK vs unquantized SHARK."""
        shark_linear = self._create_linear_layer(input_size, output_size, with_bias)
        
        # Create equivalent PyTorch layer
        torch_linear = nn.Linear(input_size, output_size, bias=with_bias)
        torch_linear.weight.data = shark_linear.theta.tensor("weight").as_torch().clone()
        if with_bias:
            torch_linear.bias.data = shark_linear.theta.tensor("bias").as_torch().clone()
        
        x = make_rand_torch([4, input_size], dtype=torch.float32) * 0.5
        
        # Test SHARK vs PyTorch
        shark_output = shark_linear(x)
        torch_output = torch_linear(x)
        torch.testing.assert_close(shark_output, torch_output, atol=1e-6, rtol=1e-6)
        
        # Test quantized SHARK vs unquantized SHARK
        # Wrap in Sequential so FX tracing treats LinearLayer as a module
        model_with_shark = nn.Sequential(shark_linear)
        quantized_model = quantize(model_with_shark)
        
        # Verify quantization actually happened
        found_quant_layer = False
        for module in quantized_model.modules():
            if isinstance(module, brevitas.nn.QuantLinear):
                found_quant_layer = True
                break
        self.assertTrue(found_quant_layer, "Expected to find qnn.QuantLinear in quantized model")
        
        # Quantized should be reasonably close to unquantized
        with torch.no_grad():
            quantized_output = quantized_model(x)
        torch.testing.assert_close(shark_output, quantized_output, atol=0.1, rtol=0.05)

    def test_shark_sequential_model_quantization(self):
        """Test quantization of a sequential model with SHARK layers."""
        linear1 = self._create_linear_layer(32, 16)
        linear2 = self._create_linear_layer(16, 8)
        
        model = nn.Sequential(linear1, nn.ReLU(), linear2, nn.Sigmoid())
        x = make_rand_torch([4, 32], dtype=torch.float32) * 0.5

        with torch.no_grad():
            reference_output = model(x)
            quantized_model = quantize(model)
            quantized_output = quantized_model(x)
        
        # Verify quantization actually happened
        quant_layer_count = 0
        for module in quantized_model.modules():
            if isinstance(module, brevitas.nn.QuantLinear):
                quant_layer_count += 1
        self.assertEqual(quant_layer_count, 2, "Expected to find 2 qnn.QuantLinear layers in quantized model")
        
        # Verify quantization preserves reasonable accuracy
        torch.testing.assert_close(reference_output, quantized_output, atol=0.2, rtol=0.1)

    def test_quantization_layer_integration(self):
        """Test QuantizationLayer with StaticScaledQuantizer and brevitas."""
        # Create a StaticScaledQuantizer for int8 quantization
        scale = torch.tensor(0.1)  # Scale factor
        quantizer = StaticScaledQuantizer(
            scale=scale,
            dtype=torch.float8_e4m3fnuz,
            name="quantizer"
        )
        
        # Create QuantizationLayer with the quantizer
        theta = Theta([quantizer])
        quant_layer = QuantizationLayer(theta, enabled=True)
        
        # Test input
        x = make_rand_torch([4, 8], dtype=torch.float32) * 2.0  # Range approx [-2, 2]
        
        # Test quantized output
        reference_output = quant_layer(x)
        
        # Should be different from input due to quantization
        self.assertFalse(torch.equal(x, reference_output.unpack().dequant()))
        
        # Test with brevitas quantization
        # Wrap in Sequential so FX tracing treats QuantizationLayer as a module
        model_with_quant = nn.Sequential(quant_layer)
        quantized_model = quantize(model_with_quant)
        
        # Verify QuantizationLayer was mapped to QuantIdentity
        found_quant_identity = False
        for module in quantized_model.modules():
            if isinstance(module, brevitas.nn.QuantIdentity):
                found_quant_identity = True
                break
        self.assertTrue(found_quant_identity, "Expected to find qnn.QuantIdentity in quantized model")
        
        # Should produce valid output
        with torch.no_grad():
            brevitas_output = quantized_model(x)
        torch.testing.assert_close(reference_output.unpack().dequant(), brevitas_output, atol=0.2, rtol=0.1)


if __name__ == "__main__":
    unittest.main()