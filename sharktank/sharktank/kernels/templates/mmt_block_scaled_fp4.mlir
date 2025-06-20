// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

{% set accum_type = "f32" %}

!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<?x?x{{k}}x!a_type>
!aexp_tensor_type = tensor<?x?x{{group0}}x{{bs}}x!a_type>
{% if batched_rhs %}
!qs_tensor_type = tensor<?x{{n}}x{{group0}}x{{bs}}xi8>
!d_tensor_type = tensor<?x{{n}}x{{group0}}x!scale_type>
!b_grouped_tensor_type = tensor<?x{{n}}x{{group0}}x{{bs}}x!a_type>
{% else %}
!qs_tensor_type = tensor<{{n}}x{{group0}}x{{bs}}xi8>
!d_tensor_type = tensor<{{n}}x{{group0}}x!scale_type>
!b_grouped_tensor_type = tensor<{{n}}x{{group0}}x{{bs}}x!a_type>
{% endif %}
!accum_tensor_type = tensor<?x?x{{n}}x!accum_type>
!c_tensor_type = tensor<?x?x{{n}}x!a_type>

module {

util.func private @sharktank_batched_block_scaled_mmt_fp4_3d_{{n}}_{{k}}_{{bs}}_{{a_type}}_{{batched_rhs}}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qs: !qs_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.0: !accum_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch0_dim = tensor.dim %a, %c0 : !a_tensor_type
  %m_dim = tensor.dim %a, %c1 : !a_tensor_type

{% if batched_rhs %}
  %b_grouped = tensor.empty(%batch0_dim) : !b_grouped_tensor_type
{% else %}
  %b_grouped = tensor.empty() : !b_grouped_tensor_type
{% endif %}

  // =============================================================================
  // FP4 DEQUANTIZATION SECTION - TEMPORARY WORKAROUND
  // TODO: Replace with native f4E2M1FN operations when MLIR backend supports it
  // =============================================================================

  // FP4 E2M1 lookup table for dequantization (from ocp_floats.py)
  %fp4_table = arith.constant dense<[
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
  ]> : tensor<16x!scale_type>

  // Dequantize: manually unpack FP4 from uint8 and lookup values
  %b_grouped_dequant = linalg.generic {
{% if batched_rhs %}
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
{% else %}
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"] }
{% endif %}
      ins(%d, %qs : !d_tensor_type, !qs_tensor_type)
      outs(%b_grouped : !b_grouped_tensor_type) {
  ^bb0(%d_element: !scale_type, %q_element: i8, %out: !a_type):
      // -------------------------------------------------------------------------
      // MANUAL FP4 CONVERSION - TO BE REPLACED WITH NATIVE OPERATIONS
      // -------------------------------------------------------------------------
      %q_element_i32 = arith.extui %q_element : i8 to i32

      // Convert to index for table lookup
      %q_element_idx = arith.index_cast %q_element_i32 : i32 to index

      // Lookup FP4 value from table
      %fp4_value = tensor.extract %fp4_table[%q_element_idx] : tensor<16x!scale_type>
      // -------------------------------------------------------------------------
      // END MANUAL FP4 CONVERSION
      // -------------------------------------------------------------------------

      // Scale the dequantized value
    {% if scale_type == a_type %}
      %q_element_scaled = arith.mulf %fp4_value, %d_element : !a_type
    {% else %}
      %d_element_ext = arith.extf %d_element : !scale_type to !a_type
      %fp4_value_ext = arith.extf %fp4_value : !scale_type to !a_type
      %q_element_scaled = arith.mulf %fp4_value_ext, %d_element_ext : !a_type
    {% endif %}
      linalg.yield %q_element_scaled : !a_type
  } -> !b_grouped_tensor_type

  // =============================================================================
  // END FP4 DEQUANTIZATION SECTION
  // =============================================================================

  // Expand %a to have the same blocked reduction structure.
  %aexp = tensor.expand_shape %a [[0], [1], [2, 3]] output_shape [%batch0_dim,%m_dim,{{group0}},{{bs}}] : !a_tensor_type into !aexp_tensor_type

  // Grouped, batch mm.
  %result_empty = tensor.empty(%batch0_dim, %m_dim) : !accum_tensor_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          // d0 = b, d1 = m, d2 = n, d3 = group0 (r), d4 = block (r)
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> ({% if batched_rhs %}d0,{% endif %} d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"] }
      ins(%aexp, %b_grouped_dequant : !aexp_tensor_type,  !b_grouped_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
  ^bb0(%a_element: !a_type, %b_element: !a_type, %out: !accum_type):
    {% if accum_type == a_type %}
      %bmm_mul = arith.mulf %a_element, %b_element : !a_type
      %bmm_accum = arith.addf %bmm_mul, %out : !a_type
    {% else %}
      %a_ext = arith.extf %a_element : !a_type to !accum_type
      %b_ext = arith.extf %b_element : !a_type to !accum_type
      %bmm_mul = arith.mulf %a_ext, %b_ext : !accum_type
      %bmm_accum = arith.addf %bmm_mul, %out : !accum_type
    {% endif %}
      linalg.yield %bmm_accum : !accum_type
  } -> !accum_tensor_type

  // Cast.
  %result_cast_empty = tensor.empty(%batch0_dim, %m_dim) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type

  util.return %result_cast : !c_tensor_type
}

}
