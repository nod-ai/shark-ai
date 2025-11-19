# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from iree.turbine.aot import ExternalTensorTrait
from pathlib import Path
from amdsharktank.types import (
    Dataset,
    DynamicFp4BlockQuantizer,
    QuantizerTensor,
    Slice,
    Theta,
)
from amdsharktank.types.layouts import BlockScaledFp4Layout
from amdsharktank.utils.testing import assert_tensor_close
from amdsharktank.utils import torch_device_equal
from amdsharktank import ops


class TestCat_BlockScaledFp4Layout:
    n_values_per_byte = 2
    block_size = 32

    @pytest.mark.parametrize(
        "dim",
        [0, 1, 2],
    )
    def test_cat_block_scaled_f4_quantization(
        self, deterministic_random_seed, dim: int
    ):
        dequantized_dtype = torch.float32
        quantizer = DynamicFp4BlockQuantizer(
            dtype=torch.float32,
            block_size=self.block_size,
            use_fe8m0_scale=True,
        )
        tensor_count = 3
        dequantized_input = [
            torch.rand([3, 5, quantizer.block_size * 7], dtype=dequantized_dtype)
            for _ in range(tensor_count)
        ]
        quantized_input = [quantizer.quantize(t) for t in dequantized_input]

        # Roundtrip the dequantization and quantization to make sure that
        # the tensor gets quantized without any loss of precision.
        dequantized_input = [
            t.to_planar().layout.dequant(dtype=dequantized_dtype)
            for t in quantized_input
        ]
        quantized_input_roundtripped = [
            quantizer.quantize(t) for t in dequantized_input
        ]
        # Make sure roundtripping is sane.
        assert_tensor_close(
            quantized_input, quantized_input_roundtripped, rtol=0, atol=0
        )

        expected_result = torch.cat(dequantized_input, dim=dim)
        actual_result = ops.cat(quantized_input, dim=dim)
        assert_tensor_close(actual_result, expected_result, rtol=0, atol=0)


class TestExtractSlice_BlockScaledFp4Layout:
    n_values_per_byte = 2
    block_size = 32

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


class TestIO:
    def test_external_tensor_trait_of_block_scaled_fp4_layout_legacy_scale_without_trailing_singleton_dimension(
        self, deterministic_random_seed, tmp_path: Path
    ):
        dtype = torch.float32
        tensor = torch.rand([2, 4], dtype=dtype)
        quantizer = DynamicFp4BlockQuantizer(
            block_size=4, use_amdsharktank_kernel=False
        )
        quantized_tensor = ops.quantize(tensor, quantizer)

        # Change to legacy shape.
        quantized_tensor.layout.d = quantized_tensor.layout.d.squeeze(-1)

        theta = Theta({"a": quantized_tensor})
        theta.rename_tensors_to_paths()
        dataset = Dataset(properties={}, root_theta=theta)
        irpa_path = tmp_path / "dataset.irpa"
        dataset.save(irpa_path)

        loaded_dataset = Dataset.load(irpa_path)
        external_tensor_trait = ExternalTensorTrait.get(
            loaded_dataset.root_theta("a").layout.d
        )
        assert external_tensor_trait != None
        assert external_tensor_trait.external_name == "a:d"


class TestMatmul:
    @pytest.mark.parametrize(
        "dtype, quantization_block_size, quantizer",
        [
            (
                torch.float32,
                32,
                DynamicFp4BlockQuantizer(
                    dtype=torch.float32, block_size=32, use_fe8m0_scale=True
                ),
            ),
            (
                torch.float32,
                32,
                DynamicFp4BlockQuantizer(
                    dtype=torch.float32, block_size=32, use_fe8m0_scale=False
                ),
            ),
        ],
    )
    def test_eager_matmul(
        self,
        deterministic_random_seed,
        dtype: torch.dtype,
        quantization_block_size: int,
        quantizer: QuantizerTensor,
    ):
        bs = 2
        m = 5
        n = 2 * quantization_block_size
        k = 3 * quantization_block_size

        lhs = torch.rand(size=[bs, m, k], dtype=dtype)
        rhs = torch.rand(size=[k, n], dtype=dtype)

        rhs_quantized = quantizer.quantize(rhs)
        lhs_quantized = quantizer.quantize(lhs)

        rhs_dequantized = rhs_quantized.unpack().dequant(dtype=dtype)
        lhs_dequantized = lhs_quantized.unpack().dequant(dtype=dtype)
        expected = torch.matmul(lhs_dequantized, rhs_dequantized)

        actual = ops.matmul(lhs_quantized, rhs_quantized)
        assert_tensor_close(actual, expected)


class TestSplit_BlockScaledFp4Layout:
    n_values_per_byte = 2
    block_size = 32

    @pytest.mark.parametrize(
        "split_size, dim",
        [
            (1, 0),
            (block_size, 2),
            (2 * block_size, -1),
            ([3, 2], 1),
            ([block_size, 2 * block_size, 4 * block_size], 2),
        ],
    )
    def test_split_block_scaled_f4_quantization(
        self, deterministic_random_seed, split_size: int | list[int], dim: int
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

        expected_result = dequantized_input.split(split_size, dim)
        actual_result = ops.split(quantized_input, split_size, dim)
        assert_tensor_close(actual_result, expected_result, rtol=0, atol=0)


class TestTo:
    block_size = 32

    @pytest.mark.parametrize(
        "source_device, target_device",
        [
            (torch.device("cpu"), torch.device("cpu")),
            pytest.param(
                torch.device("cuda"),
                torch.device("cpu"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Needs CUDA/HIP device."
                ),
            ),
            pytest.param(
                torch.device("cpu"),
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Needs CUDA/HIP device."
                ),
            ),
        ],
    )
    def test_to_device_BlockScaledFp4Layout(
        self,
        deterministic_random_seed,
        source_device: torch.device,
        target_device: torch.device,
    ):
        dequantized_dtype = torch.float32
        quantizer = DynamicFp4BlockQuantizer(
            dtype=torch.float32,
            block_size=self.block_size,
            use_fe8m0_scale=True,
        )
        dequantized_input = torch.zeros(
            [1, 2, quantizer.block_size * 3],
            dtype=dequantized_dtype,
            device=source_device,
        )
        quantized_input = quantizer.quantize(dequantized_input)

        method_result = quantized_input.to(device=target_device)
        layout: BlockScaledFp4Layout = method_result.to_planar().layout
        for plane_tensor in layout.planes.values():
            assert torch_device_equal(plane_tensor.device, target_device)

        ops_result = ops.to(quantized_input, device=target_device)
        layout: BlockScaledFp4Layout = ops_result.to_planar().layout
        for plane_tensor in layout.planes.values():
            assert torch_device_equal(plane_tensor.device, target_device)
