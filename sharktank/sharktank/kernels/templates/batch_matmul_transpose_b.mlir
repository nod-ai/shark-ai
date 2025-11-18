// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!dtype = {{dtype}}
!a_tensor_type = {{a_asm_type}}
!b_tensor_type = {{b_asm_type}}
!c_tensor_type = {{c_asm_type}}
!c_dynamic_tensor_type = tensor<?x?x?x!dtype>

module {

util.func private @amdsharktank_batch_matmul_transpose_b_{{spec_sig}}(
    %a: !a_tensor_type, %b: !b_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant {{cst_zero}}: !dtype
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %batch_dim = tensor.dim %a, %c0 : !a_tensor_type  // b, m, k
  %m_dim = tensor.dim %a, %c1 : !a_tensor_type  // b, m, k
  %n_dim = tensor.dim %b, %c1 : !b_tensor_type  // b, n, k
  %result_empty_dynamic = tensor.empty(%batch_dim, %m_dim, %n_dim) : !c_dynamic_tensor_type
  %result_empty = tensor.cast %result_empty_dynamic : !c_dynamic_tensor_type to !c_tensor_type
  %result_fill = linalg.fill ins(%zero: !dtype) outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.batch_matmul
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ]
    ins(%a, %b: !a_tensor_type, !b_tensor_type) outs(%result_fill: !c_tensor_type) -> !c_tensor_type
  util.return %result : !c_tensor_type
}

}
