// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!elem_type = {{dtype}}
!tensor_type = tensor<{% for i in range(rank) %}?x{% endfor %}!elem_type>
!x_tensor_type = tensor<{{xshape}}x!elem_type>
!y_tensor_type = tensor<{{yshape}}x!elem_type>
!x_collapsed_type = tensor<{% for i in range(rank - x_size) %}?x{% endfor %}!elem_type>
!y_collapsed_type = tensor<{% for i in range(rank - y_size) %}?x{% endfor %}!elem_type>

module {

util.func private @sharktank_elementwise_tensor_tensor_{{op_str}}_{{rank}}_{{dtype}}_{{broadcast_str}}(
    %x: !x_tensor_type, %y: !y_tensor_type)
    -> !tensor_type {
  // Get the dimensions after broadcast
  {% for i in range(rank) %}
  %k{{i}} = arith.constant {{i}} : index
  %xdim{{i}} = tensor.dim %x, %k{{i}}: !x_tensor_type
  %ydim{{i}} = tensor.dim %y, %k{{i}}: !y_tensor_type
  %xint{{i}} = arith.index_cast %xdim{{i}} : index to i32
  %yint{{i}} = arith.index_cast %ydim{{i}} : index to i32
  %max{{i}} = arith.maxsi %xint{{i}}, %yint{{i}} : i32
  %dim{{i}} = arith.index_cast %max{{i}} : i32 to index
  {% endfor %}
  %x_empty = tensor.empty({% for i in range(rank - 1) %}%dim{{i}}, {% endfor %}%dim{{rank - 1}}) : !tensor_type
  %y_empty = tensor.empty({% for i in range(rank - 1) %}%dim{{i}}, {% endfor %}%dim{{rank - 1}}) : !tensor_type
  %x_collapsed = tensor.collapse_shape %x {{x_reassociation}}: !x_tensor_type into !x_collapsed_type
  %y_collapsed = tensor.collapse_shape %y {{y_reassociation}}: !y_tensor_type into !y_collapsed_type
  %x_broadcast = linalg.broadcast ins(%x_collapsed: !x_collapsed_type) outs(%x_empty: !tensor_type) dimensions = [{{x_broadcast}}]
  %y_broadcast = linalg.broadcast ins(%y_collapsed: !y_collapsed_type) outs(%y_empty: !tensor_type) dimensions = [{{y_broadcast}}]
  %result_empty = tensor.empty({% for i in range(rank - 1) %}%dim{{i}}, {% endfor %}%dim{{rank - 1}}) : !tensor_type
  %result = linalg.generic {
      indexing_maps = [
          affine_map<({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}}) -> ({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}})>,
          affine_map<({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}}) -> ({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}})>,
          affine_map<({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}}) -> ({% for i in range(rank - 1) %}d{{i}}, {% endfor %}d{{rank - 1}})>
      ],
      iterator_types = [
        {% for i in range(rank - 1) %}"parallel", {% endfor %}"parallel"
      ] }
      ins(%x_broadcast, %y_broadcast : !tensor_type, !tensor_type)
      outs(%result_empty : !tensor_type) {
  ^bb0(%x_element: !elem_type, %y_element: !elem_type, %out: !elem_type):
      %elem_result = arith.{{op_str}}f %x_element, %y_element : !elem_type
      linalg.yield %elem_result : !elem_type
  } -> !tensor_type

  util.return %result : !tensor_type
}

}
