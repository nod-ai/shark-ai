// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!input_tensor_type = {{input_tensor_type}}
!table_tensor_type = {{table_tensor_type}}

module {

util.func private @amdsharktank_rotary_embedding_{{bs}}_{{sl}}_{{heads}}_{{dims}}_{{dtype}}(%input: !input_tensor_type, %table: !table_tensor_type) -> !input_tensor_type {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index


  %d0 = tensor.dim %input, %c0 : !input_tensor_type
  %d1 = tensor.dim %input, %c1 : !input_tensor_type
  %d2 = tensor.dim %input, %c2 : !input_tensor_type
  %d3 = tensor.dim %input, %c3 : !input_tensor_type

  %empty_dyn = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?x{{dtype}}>
  %empty = tensor.cast %empty_dyn : tensor<?x?x?x?x{{dtype}}> to {{input_tensor_type}}

  %result = linalg.generic {
      indexing_maps = [
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                       ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%table : !table_tensor_type )
      outs(%empty : !input_tensor_type) {
    ^bb0(%b0 : {{dtype}} , %b1 : {{dtype}}):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = linalg.index 3 : index
      %div = arith.divui %3, %c2 : index
      %mod = arith.remui %3, %c2 : index
      %a_cosb = math.cos %b0 : {{dtype}}
      %a_sinb = math.sin %b0 : {{dtype}}
      %real_index = arith.muli %div, %c2 : index
      %imag_index = arith.addi %real_index, %c1 : index
      %real = tensor.extract %input[%0, %1, %2, %real_index] : !input_tensor_type
      %imag = tensor.extract %input[%0, %1, %2, %imag_index] : !input_tensor_type
      %cmp = arith.cmpi eq, %mod, %c0 : index
      %real_t0 = arith.mulf %real, %a_cosb : {{dtype}}
      %real_t1 = arith.mulf %imag, %a_sinb : {{dtype}}
      %real_t2 = arith.subf %real_t0, %real_t1 : {{dtype}}
      %imag_t0 = arith.mulf %imag, %a_cosb : {{dtype}}
      %imag_t1 = arith.mulf %real, %a_sinb : {{dtype}}
      %imag_t2 = arith.addf %imag_t0, %imag_t1 : {{dtype}}
      %val = arith.select %cmp, %real_t2, %imag_t2 : {{dtype}}
      linalg.yield %val : {{dtype}}
  } -> !input_tensor_type

  util.return %result : !input_tensor_type
}

}
