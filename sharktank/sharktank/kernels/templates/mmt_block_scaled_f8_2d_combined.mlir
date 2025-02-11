// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

{% set accum_type = "bf16" %}

!lowp_type = f8E5M2FNUZ
!a_type = {{a_type}}
!c_type = {{c_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<{{m}}x{{k}}x!a_type>
!b_tensor_type = tensor<{{k}}x{{n}}x!a_type>
!c_tensor_type = tensor<{{m}}x{{n}}x!c_type>
!as_tensor_type = tensor<{{num_blocks_m}}x{{num_blocks_k}}x!scale_type>
!bs_tensor_type = tensor<{{num_blocks_k}}x{{num_blocks_n}}x!scale_type>
!accum_tensor_type = tensor<{{m}}x{{n}}x!accum_type>

module {

util.func private @sharktank_mmt_block_scaled_f8_2d_{{m}}_{{n}}_{{k}}_{{a_type}}(
    %a: !a_tensor_type, %b: !b_tensor_type, %as: !as_tensor_type, %bs: !bs_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.0: !accum_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %a, %c0 : !a_tensor_type
  %n = tensor.dim %b, %c1 : !b_tensor_type

  %result_empty = tensor.empty(%m, %n) : !accum_tensor_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          // d0 = m, d1 = n, d2 = k
          // a
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          // b
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          // as
          affine_map<(d0, d1, d2) -> (d0 floordiv {{block_size_m}}, d2 floordiv {{block_size_k}})>,
          // bs
          affine_map<(d0, d1, d2) -> (d2 floordiv {{block_size_k}}, d1 floordiv {{block_size_n}})>,
          // out
          affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"] }
      ins(%a, %b, %as, %bs : !a_tensor_type, !b_tensor_type, !as_tensor_type, !bs_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
  ^bb0(%a_element: !a_type, %b_element: !a_type, %as_element: !scale_type, %bs_element: !scale_type, %out: !accum_type):
      %mm_mul = arith.mulf %a_element, %b_element : !a_type
      %s_mul = arith.mulf %as_element, %bs_element : !scale_type
    {% if accum_type == a_type && accum_type == scale_type %}
      %scaled_mm = arith.mulf %mm_mul, %s_mul : !accum_type
    {% else if accum_type == a_type && accum_type != scale_type %}
      %s_mul_ext = arith.extf %s_mul : !scale_type to !accum_type
      %scaled_mm = arith.mulf %mm_mul, %s_mul_ext : !accum_type
    {% else if accum_type != a_type && accum_type == scale_type %}
      %mm_mul_ext = arith.extf %mm_mul : !a_type to !accum_type
      %scaled_mm = arith.mulf %mm_mul_ext, %s_mul : !accum_type
    {% else if accum_type != a_type && accum_type != scale_type %}
      %mm_mul_ext = arith.extf %mm_mul : !a_type to !accum_type
      %s_mul_ext = arith.extf %s_mul : !scale_type to !accum_type
      %scaled_mm = arith.mulf %mm_mul_ext, %s_mul_ext : !accum_type
    {% endif %}
      %mm_accum = arith.addf %scaled_mul, %out : !accum_type
      linalg.yield %mm_accum : !accum_type
  } -> !accum_tensor_type

  // Cast.
  %result_cast_empty = tensor.empty(%m, %n) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
  util.return %result_cast : !c_tensor_type
}

}
