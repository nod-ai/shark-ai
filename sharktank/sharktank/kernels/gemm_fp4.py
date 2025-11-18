# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *
from amdsharktank.kernels.mlir_kernel import *
from iree.compiler.ir import (
    Module,
    Context,
)
import torch


__all__ = [
    "iree_mxfp4_bmm",
]

B2 = DynDim.B
M2 = DynDim.M
N2 = StaticDim.N
HALF_K = StaticDim.HALF_K
K_OVER_THIRTYTWO = StaticDim.K_OVER_THIRTYTWO

U8 = Dtype.U8(torch.uint8)
F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)


@mlir_kernel(
    inputs=(
        MLIRTensor[B2, M2, HALF_K, U8],
        MLIRTensor[B2, M2, K_OVER_THIRTYTWO, U8],
        MLIRTensor[N2, HALF_K, U8],
        MLIRTensor[N2, K_OVER_THIRTYTWO, U8],
    ),
    results=(MLIRTensor[B2, M2, N2, F16],),
)
def iree_mxfp4_bmm(x, x_scales, w_t, w_scales, result=None):
    mlir = """
    #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
    #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
    #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
    #map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
    #map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
    #map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
    util.func private @{{kernel_name}}(%x : !x, %x_scales : !x_scales, %w_t: !w_t, %w_scales: !w_scales) -> !result {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index

      %cast = tensor.cast %x : !x to tensor<?x?x?xi8>
      %cast_0 = tensor.cast %w_t : !w_t to tensor<?x?xi8>
      %cast_1 = tensor.cast %x_scales : !x_scales to tensor<?x?x?xi8>
      %cast_2 = tensor.cast %w_scales : !w_scales to tensor<?x?xi8>
      // %cast_3 = tensor.cast %res : !res to tensor<?x?x?xf16>

      %B = tensor.dim %cast, %c0 : tensor<?x?x?xi8>
      %M = tensor.dim %cast, %c1 : tensor<?x?x?xi8>
      %K_u8 = tensor.dim %cast, %c2 : tensor<?x?x?xi8>
      %Ks = tensor.dim %cast_1, %c2 : tensor<?x?x?xi8>
      %dim_7 = tensor.dim %cast_0, %c0 : tensor<?x?xi8>

      %K_f4 = arith.muli %K_u8, %c2 overflow<nsw, nuw>: index

      %0 = iree_tensor_ext.bitcast %cast : tensor<?x?x?xi8>{% raw %}{%B, %M, %K_u8}{% endraw %} -> tensor<?x?x?xf4E2M1FN>{% raw %}{%B, %M, %K_f4}{% endraw %}
      %1 = iree_tensor_ext.bitcast %cast_0 : tensor<?x?xi8>{% raw %}{%dim_7, %K_u8}{% endraw %} -> tensor<?x?xf4E2M1FN>{% raw %}{%dim_7, %K_f4}{% endraw %}
      %2 = iree_tensor_ext.bitcast %cast_1 : tensor<?x?x?xi8>{% raw %}{%B, %M, %Ks}{% endraw %} -> tensor<?x?x?xf8E8M0FNU>{% raw %}{%B, %M, %Ks}{% endraw %}
      %3 = iree_tensor_ext.bitcast %cast_2 : tensor<?x?xi8>{% raw %}{%dim_7, %Ks}{% endraw %} -> tensor<?x?xf8E8M0FNU>{% raw %}{%dim_7, %Ks}{% endraw %}

      %4 = arith.divui %K_f4, %c32 : index
      %5 = arith.divui %Ks, %c32 : index
      %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [%B, %M, %4, 32] : tensor<?x?x?xf4E2M1FN> into tensor<?x?x?x32xf4E2M1FN>
      %expanded_9 = tensor.expand_shape %1 [[0], [1, 2]] output_shape [%dim_7, %4, 32] : tensor<?x?xf4E2M1FN> into tensor<?x?x32xf4E2M1FN>
      %cst = arith.constant 0.000000e+00 : f32
      %6 = tensor.empty(%B, %M, %dim_7) : tensor<?x?x?xf32>
      %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%expanded, %expanded_9, %2, %3 : tensor<?x?x?x32xf4E2M1FN>, tensor<?x?x32xf4E2M1FN>, tensor<?x?x?xf8E8M0FNU>, tensor<?x?xf8E8M0FNU>) outs(%7 : tensor<?x?x?xf32>) {
      ^bb0(%in: f4E2M1FN, %in_10: f4E2M1FN, %in_11: f8E8M0FNU, %in_12: f8E8M0FNU, %out: f32):
        %10 = arith.scaling_extf %in, %in_11 : f4E2M1FN, f8E8M0FNU to f32
        %11 = arith.scaling_extf %in_10, %in_12 : f4E2M1FN, f8E8M0FNU to f32
        %12 = arith.mulf %10, %11 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      } -> tensor<?x?x?xf32>
      %11 = tensor.empty(%B, %M, %dim_7) : tensor<?x?x?xf16>
      %9 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<?x?x?xf32>) outs(%11 : tensor<?x?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %10 = arith.truncf %in : f32 to f16
        linalg.yield %10 : f16
      } -> tensor<?x?x?xf16>
      %out = tensor.cast %9 : tensor<?x?x?xf16> to !result
      util.return %out : !result
    }
    }
    """
    return MLIRSpec(mlir)
