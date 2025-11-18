// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>

!accum_type = {{accum_type}}
!inputs_asm_type = {{inputs_asm_type}}
!weights_asm_type = {{weights_asm_type}}
!bias_asm_type = {{bias_asm_type}}
!result_asm_type = {{result_asm_type}}
!dynamic_result_asm_type = tensor<?x?x?x?x{{accum_type}}>

module {

util.func private @amdsharktank_conv_2d_nchw_fchw_{{spec_sig}}
  (%input_pad: !inputs_asm_type, %weights: !weights_asm_type, %bias: !bias_asm_type)
    -> !result_asm_type {
  %zero = arith.constant {{zero}}: !accum_type
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %rN = tensor.dim %input_pad, %c0 : !inputs_asm_type
  %rC = tensor.dim %weights, %c0 : !weights_asm_type
  %rH = arith.constant {{H_out}} : index
  %rW = arith.constant {{W_out}} : index
  %result_empty_dynamic = tensor.empty(%rN, %rC, %rH, %rW) : !dynamic_result_asm_type
  %result_empty = tensor.cast %result_empty_dynamic : !dynamic_result_asm_type to !result_asm_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !result_asm_type) -> !result_asm_type
  %result = linalg.conv_2d_nchw_fchw
    {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>,
     strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>}
    ins(%input_pad, %weights: !inputs_asm_type, !weights_asm_type)
    outs(%result_fill: !result_asm_type) -> !result_asm_type
  %result_biased = linalg.generic {
      indexing_maps = [#map0, #map1, #map0],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%result, %bias : !result_asm_type, !bias_asm_type)
    outs(%result_empty : !result_asm_type) {
    ^bb0(%in: !accum_type, %in_1: !accum_type, %out: !accum_type):
      %add = {{add_op}} %in, %in_1 : !accum_type
      linalg.yield %add : !accum_type
    } -> !result_asm_type
  util.return %result_biased : !result_asm_type
}

}
