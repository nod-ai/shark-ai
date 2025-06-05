// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!input_tensor_type = tensor<?x?x{{dtype}}>
!indices_tensor_type = tensor<?x?xi32>
!values_tensor_type = tensor<?x{{k}}x{{dtype}}>
!indices_out_tensor_type = tensor<?x{{k}}xi32>

module {
  util.func private @sharktank_topk_{{k}}_{{dtype}}(%arg0: !input_tensor_type) -> (!values_tensor_type, !indices_out_tensor_type) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.402820e+38 : {{dtype}}  // Minimum float32 value

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index


    %dim0 = tensor.dim %arg0, %c0 : !input_tensor_type
    %dim1 = tensor.dim %arg0, %c1 : !input_tensor_type
    %0 = tensor.empty(%dim0, %dim1) : !indices_tensor_type
    %1 = tensor.empty(%dim0) : !values_tensor_type
    %2 = tensor.empty(%dim0) : !indices_out_tensor_type
    %3 = linalg.fill ins(%cst : {{dtype}}) outs(%1 : !values_tensor_type) -> !values_tensor_type
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : !indices_out_tensor_type) -> !indices_out_tensor_type
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%0 : !indices_tensor_type) {
    ^bb0(%out: i32):
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      linalg.yield %8 : i32
    } -> !indices_tensor_type
    %6:2 = iree_linalg_ext.topk dimension(1) ins(%arg0, %5 : !input_tensor_type, !indices_tensor_type) outs(%3, %4 : !values_tensor_type, !indices_out_tensor_type) {
    ^bb0(%arg1: {{dtype}}, %arg2: {{dtype}}):
      // Sort in descending order like PyTorch
      %7 = arith.cmpf ogt, %arg1, %arg2 : {{dtype}}
      iree_linalg_ext.yield %7 : i1
    } -> !values_tensor_type, !indices_out_tensor_type
    util.return %6#0, %6#1 : !values_tensor_type, !indices_out_tensor_type
  }
}
