# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *
from amdsharktank.kernels.mlir_kernel import *

import torch

__all__ = [
    "flash_attention",
    "masked_flash_attention",
]

BATCH = DynDim.BATCH
NUM_HEADS = DynDim.NUM_HEADS
M = DynDim.M
K1 = StaticDim.K1
K2 = DynDim.K2
N = StaticDim.N

I_DTYPE = Dtype.I_DTYPE
M_DTYPE = Dtype.M_DTYPE
S_DTYPE = Dtype.S_DTYPE
O_DTYPE = Dtype.O_DTYPE(torch.float32)


@mlir_kernel(
    inputs=(
        MLIRTensor[BATCH, NUM_HEADS, M, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, N, I_DTYPE],
        MLIRTensor[S_DTYPE],
    ),
    results=(MLIRTensor[BATCH, NUM_HEADS, M, N, O_DTYPE],),
)
def flash_attention(q, k, v, scale, result=None):
    mlir = """
    module {
    util.func private @{{kernel_name}}(%q : !q, %k : !k, %v: !v, %scale: !scale) -> !result {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %batch = tensor.dim %q, %c0 : !q
      %num_heads = tensor.dim %q, %c1 : !q
      %m = tensor.dim %q, %c2 : !q

      %empty = tensor.empty(%batch, %num_heads, %m) : !result

      %s_c = tensor.extract %scale[] : !scale

      %result = iree_linalg_ext.attention {
        indexing_maps = [
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, N)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> ()>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, N)>
        ]
      }
      ins(%q, %k, %v, %s_c : !q, !k, !v, !scale_dtype)
      outs(%empty : !result) {
        ^bb0(%score : f32):
          iree_linalg_ext.yield %score : f32
      } -> !result

      util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)


@mlir_kernel(
    inputs=(
        MLIRTensor[BATCH, NUM_HEADS, M, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, N, I_DTYPE],
        MLIRTensor[M, K2, M_DTYPE],
        MLIRTensor[S_DTYPE],
    ),
    results=(MLIRTensor[BATCH, NUM_HEADS, M, N, O_DTYPE],),
)
def masked_flash_attention(q, k, v, mask, scale, result=None):
    mlir = """
    module {
    util.func private @{{kernel_name}}(%q : !q, %k : !k, %v: !v, %mask : !mask, %scale: !scale) -> !result {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %batch = tensor.dim %q, %c0 : !q
      %num_heads = tensor.dim %q, %c1 : !q
      %m = tensor.dim %q, %c2 : !q

      %empty = tensor.empty(%batch, %num_heads, %m) : !result

      %s_c = tensor.extract %scale[] : !scale

      %result = iree_linalg_ext.attention {
        indexing_maps = [
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, N)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> ()>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (M, K2)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, N)>
        ]
      }
      ins(%q, %k, %v, %s_c, %mask : !q, !k, !v, !scale_dtype, !mask)
      outs(%empty : !result) {
        ^bb0(%score : f32):
          iree_linalg_ext.yield %score : f32
      } -> !result

      util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)
