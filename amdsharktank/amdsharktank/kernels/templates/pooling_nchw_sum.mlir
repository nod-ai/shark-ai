// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!accum_dtype = {{accum_type}}
!inputs_asm_type = {{inputs_asm_type}}
!result_asm_type = {{result_asm_type}}
!dynamic_result_asm_type = tensor<?x?x?x?x!accum_dtype>
!weights_tensor_type = tensor<{{ks_H}}x{{ks_W}}x!accum_dtype>

module {

util.func private @amdsharktank_pooling_nchw_sum_{{spec_sig}} (
    %input_pad: !inputs_asm_type)
    -> !result_asm_type {
  %zero = arith.constant 0: !accum_dtype
  %weights = tensor.empty() : !weights_tensor_type
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index

  %rN = tensor.dim %input_pad, %c0 : !inputs_asm_type
  %rC = tensor.dim %input_pad, %c1 : !inputs_asm_type
  %rH = arith.constant {{H_out}} : index
  %rW = arith.constant {{W_out}} : index
  %result_empty_dynamic = tensor.empty(%rN, %rC, %rH, %rW) : !dynamic_result_asm_type
  %result_empty = tensor.cast %result_empty_dynamic : !dynamic_result_asm_type to !result_asm_type
  %result_fill = linalg.fill ins(%zero: !accum_dtype) outs(%result_empty: !result_asm_type) -> !result_asm_type
  %result = linalg.pooling_nchw_sum
    {dilations = dense<[{{dilations_H}}, {{dilations_W}}]> : tensor<2xi64>,
     strides = dense<[{{strides_H}}, {{strides_W}}]> : tensor<2xi64>}
    ins(%input_pad, %weights: !inputs_asm_type, !weights_tensor_type)
    outs(%result_fill: !result_asm_type) -> !result_asm_type
  util.return %result : !result_asm_type
}

}
