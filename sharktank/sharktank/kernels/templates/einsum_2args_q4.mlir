// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

{% set accum_type = "f32" %}

!lowp_type = i4
!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<{% for i in range(a_size) %}?x{% endfor %}!a_type>
!qs_raw_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}{{bs_i8}}xi8>
!qs_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}{{bs}}x!lowp_type>
!d_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}1x!scale_type>
!m_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}1x!scale_type>
!accum_tensor_type = tensor<{% for i in range(c_size) %}?x{% endfor %}!accum_type>
!c_tensor_type = tensor<{% for i in range(c_size) %}?x{% endfor %}!a_type>
!b_grouped_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}{{bs}}x!a_type>
!b_tensor_type = tensor<{% for i in range(b_size) %}?x{% endfor %}!a_type>

module {

util.func private @amdsharktank_einsum_2args_q4_{{es_name}}_{{bs}}_{{a_type}}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qs_raw: !qs_raw_tensor_type, %m: !m_tensor_type)
    -> !c_tensor_type {
  %debug = tensor.empty() : tensor<1xf32>
  %zero = arith.constant 0.0: !accum_type
  {% for i in range(a_size) %}
  %k{{i}} = arith.constant {{i}} : index
  {% endfor %}
  {% for i in range(a_size, b_size) %}
  %k{{i}} = arith.constant {{i}} : index
  {% endfor %}
  {% for i in range(a_size) %}
  %a{{i}} = tensor.dim %a, %k{{i}}: !a_tensor_type
  {% endfor %}
  {% for i in range(b_size) %}
  %b{{i}} = tensor.dim %qs_raw, %k{{i}}: !qs_raw_tensor_type
  {% endfor %}
  %bs = arith.constant {{bs}} : index
  %b_unblocked_dim = arith.muli %b{{b_size-1}}, %bs : index

  //%qs = flow.tensor.bitcast %qs_raw : !qs_raw_tensor_type -> !qs_tensor_type
  %qs = flow.tensor.bitcast %qs_raw : !qs_raw_tensor_type{{"{"}}{% for i in range(b_size-1) %}%b{{i}},{% endfor %}%b{{b_size-1}}{{"}"}} -> !qs_tensor_type{{"{"}}{% for i in range(b_size-1) %}%b{{i}},{% endfor %}%b{{b_size-1}}{{"}"}}

  // Dequantize.
  %b_grouped = tensor.empty({% for i in range(b_size-1) %}%b{{i}},{% endfor %}%b{{b_size-1}}) : !b_grouped_tensor_type
  %b_grouped_dequant = linalg.generic {
      indexing_maps = [
          {{dequant_indexing_maps}}],
      iterator_types = [{{dequant_iterator_types}}] }
      ins(%d, %m, %qs : !d_tensor_type, !m_tensor_type, !qs_tensor_type)
      outs(%b_grouped : !b_grouped_tensor_type) {
  ^bb0(%d_element: !scale_type, %m_element: !scale_type, %q_element: !lowp_type, %out: !a_type):
      %q_element_ext = arith.extui %q_element : !lowp_type to i32
      %q_element_fp = arith.uitofp %q_element_ext : i32 to !a_type
    {% if scale_type == a_type %}
      %q_element_scaled = arith.mulf %q_element_fp, %d_element : !a_type
      %q_element_offset = arith.addf %q_element_scaled, %m_element : !a_type
    {% else %}
      %d_element_ext = arith.extf %d_element : !scale_type to !a_type
      %m_element_ext = arith.extf %m_element : !scale_type to !a_type
      %q_element_scaled = arith.mulf %q_element_fp, %d_element_ext : !a_type
      %q_element_offset = arith.addf %q_element_scaled, %m_element_ext : !a_type
    {% endif %}
      linalg.yield %q_element_offset : !a_type
  } -> !b_grouped_tensor_type

  // Collapse %b to the same unblocked structure.
  %b_unblocked = tensor.collapse_shape %b_grouped_dequant [{% for i in range(b_size-1) %}[{{i}}], {% endfor %}[{{b_size-1}}, {{b_size}}]] : !b_grouped_tensor_type into !b_tensor_type

  // Einsum
  %result_empty = tensor.empty({{out_dyn_dim_size_str}}) : !accum_tensor_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          {{einsum_indexing_maps}}],
      iterator_types = [{{einsum_iterator_types}}] }
      ins(%a, %b_unblocked : !a_tensor_type,  !b_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
  ^bb0(%a_element: !a_type, %b_element: !a_type, %out: !accum_type):
      %bmm_mul = arith.mulf %a_element, %b_element : !a_type
    {% if accum_type == a_type %}
      %bmm_accum = arith.addf %bmm_mul, %out : !a_type
    {% else %}
      %bmm_mul_ext = arith.extf %bmm_mul : !a_type to !accum_type
      %bmm_accum = arith.addf %bmm_mul_ext, %out : !accum_type
    {% endif %}
      linalg.yield %bmm_accum : !accum_type
  } -> !accum_tensor_type

  // Cast.
  %result_cast_empty = tensor.empty({{out_dyn_dim_size_str}}) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type

  //iree_input.tensor.trace "foobar" = [%a : !a_tensor_type, %d : !d_tensor_type, %qs_raw: !qs_raw_tensor_type, %m: !m_tensor_type, %b_grouped_dequant: !b_grouped_tensor_type]
  util.return %result_cast : !c_tensor_type
}

}
