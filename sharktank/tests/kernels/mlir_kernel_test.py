# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import unittest

from sharktank.kernels.mlir_kernel import *

N = DynDim.N
M = StaticDim.M
K = DynDim.K

S = Dtype.S
I64 = Dtype.I64


@mlir_kernel(
    inputs=(MLIRTensor[N, M, S], MLIRTensor[N, I64]),
    results=(MLIRTensor[N, M, S],),
)
def sharktank_gather(source, indices, result=None):
    mlir = """
    module {
    util.func @{{kernel_name}}(%source: !source, %indices: !indices) -> !result {
      %c0 = arith.constant 0 : index
      %n_dim = tensor.dim %source, %c0 : !source
      %empty = tensor.empty(%n_dim) : !result
      %result = linalg.generic {
        indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%indices : !indices)
        outs(%empty : !result) {
        ^bb0(%in: !indices_dtype, %o: !source_dtype):
          %n = arith.index_cast %in : !indices_dtype to index
          %m = linalg.index 1 : index
          %extracted = tensor.extract %source[%n, %m] : !source
          linalg.yield %extracted : !source_dtype
        } -> !result
        util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)


@mlir_kernel(
    inputs=(MLIRTensor[N, M, S], MLIRTensor[I64]),
    results=(MLIRTensor[N, DynDim.K, S], MLIRTensor[N, DynDim.K, I64]),
)
def sharktank_topk(input, k_tensor, values=None, indices=None):
    mlir = """
    module {
    util.func @{{kernel_name}}(%input: !input, %k: !k_tensor) -> (!values, !indices) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i64 = arith.constant 0 : i64
      %c_min = arith.constant -32768.0 : f16

      // Get input dimensions
      %n_dim = tensor.dim %input, %c0 : !input
      %m_dim = tensor.dim %input, %c1 : !input

      // Extract k value and create output tensors
      %k_val = tensor.extract %k[%c0] : !k_tensor
      %k_dim = arith.index_cast %k_val : i64 to index

      // Initialize output tensors
      %empty_values = tensor.empty(%n_dim, %k_dim) : !values
      %empty_indices = tensor.empty(%n_dim, %k_dim) : !indices

      // Initialize with minimum value
      %init_values = linalg.fill ins(%c_min : f16) outs(%empty_values : !values) -> !values
      %init_indices = linalg.fill ins(%c0_i64 : i64) outs(%empty_indices : !indices) -> !indices

      // For each row, find top-k values and their indices
      %result_values, %result_indices = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,  // input: [n, m]
          affine_map<(d0, d1, d2) -> (d0, d1)>,  // values output: [n, k]
          affine_map<(d0, d1, d2) -> (d0, d1)>   // indices output: [n, k]
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%input : !input) outs(%init_values, %init_indices : !values, !indices) {
      ^bb0(%in: !input_dtype, %val: !values_dtype, %idx: !indices_dtype):
        // Get current column index
        %col_idx = linalg.index 2 : index
        %col_idx_i64 = arith.index_cast %col_idx : index to i64

        // Get current output position
        %out_pos = linalg.index 1 : index

        // Compare with current value
        %is_greater = arith.cmpf "ogt", %in, %val : !input_dtype

        // Select new value and index if greater
        %new_val = arith.select %is_greater, %in, %val : !values_dtype
        %new_idx = arith.select %is_greater, %col_idx_i64, %idx : !indices_dtype

        linalg.yield %new_val, %new_idx : !values_dtype, !indices_dtype
      } -> (!values, !indices)

      util.return %result_values, %result_indices : !values, !indices
    }
    }
    """
    return MLIRSpec(mlir)


class mlir_kernel_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)

    def test_mlir_kernel(self):
        source = torch.randn([64, 32]).to(torch.float16)
        indices = torch.tensor([3, 7, 54])
        out = sharktank_gather(source, indices)
        torch.testing.assert_close(source[3], out[0])
        torch.testing.assert_close(source[7], out[1])
        torch.testing.assert_close(source[54], out[2])

    def test_topk_kernel(self):
        # Create a random input tensor
        input_tensor = torch.randn([4, 8]).to(torch.float16)
        print("input_tensor", input_tensor)
        k = 4

        # Create k tensor (just a tensor with the k value)
        k_tensor = torch.ones(k, dtype=torch.int64)

        print("k_tensor", k_tensor)

        # Get top-k values and indices using our kernel
        values, indices = sharktank_topk(input_tensor, k_tensor)

        # Get top-k values and indices using PyTorch for comparison
        torch_values, torch_indices = torch.topk(input_tensor, k=k, dim=-1, sorted=True)

        print("values", values)
        print("torch_values", torch_values)
        print("indices", indices)
        print("torch_indices", torch_indices)
        # Compare results
        torch.testing.assert_close(values, torch_values)
        torch.testing.assert_close(indices, torch_indices)
