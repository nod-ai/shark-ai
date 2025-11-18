// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!a_type = {{a_type}}
!bT_type = {{bT_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<?x{{k}}x!a_type>
!bT_tensor_type = tensor<{{n}}x{{k}}x!bT_type>
!accum_tensor_type = tensor<?x{{n}}x!accum_type>
!c_tensor_type = tensor<?x{{n}}x!a_type>

module {

util.func private @amdsharktank_mmtfp_2d_{{n}}_{{k}}_{{a_type}}{{bT_type}}{{accum_type}}(
    %a: !a_tensor_type, %bT: !bT_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.000000e+00 : !accum_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %a, %c0 : !a_tensor_type
  %result_empty = tensor.empty(%m) : !accum_tensor_type
  %result_init = linalg.fill
    ins(%zero : !accum_type)
    outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result_accum = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins (%a, %bT: !a_tensor_type, !bT_tensor_type)
    outs (%result_init: !accum_tensor_type) -> !accum_tensor_type
  %result_cast_empty = tensor.empty(%m) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result_accum : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
  util.return %result_cast : !c_tensor_type
}

}
