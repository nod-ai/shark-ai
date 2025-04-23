// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!input_tensor_type = {{input_tensor_type}}
!table_tensor_type = {{table_tensor_type}}
!output_tensor_type = {{output_tensor_type}}

module {

util.func private @{{func_name}}(%input: !input_tensor_type, %table: !table_tensor_type) -> !output_tensor_type {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index


  %d0 = tensor.dim %input, %c0 : !input_tensor_type
  %d1 = tensor.dim %input, %c1 : !input_tensor_type
  %d2 = tensor.dim %input, %c2 : !input_tensor_type
  %d3 = tensor.dim %input, %c3 : !input_tensor_type
  %half_d3 = arith.divui %d3, %c2 : index

  %input_dynamic = tensor.cast %input : !input_tensor_type to tensor<?x?x?x?x{{input_dtype}}>
  %input_complex = flow.tensor.bitcast %input_dynamic :
                     tensor<?x?x?x?x{{input_dtype}}>{{'{%d0, %d1, %d2, %d3}'}}
                     -> tensor<?x?x?x?xcomplex<{{input_dtype}}>>{{'{%d0, %d1, %d2, %half_d3}'}}

  %empty_dyn = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?x{{dtype}}>
  %empty = tensor.cast %empty_dyn : tensor<?x?x?x?x{{dtype}}> to {{output_tensor_type}}

  %result = linalg.generic {
      indexing_maps = [
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 floordiv 2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                       ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%table, %input_complex : !table_tensor_type, tensor<?x?x?x?xcomplex<{{input_dtype}}>>)
      outs(%empty : !output_tensor_type) {
      ^bb0(%b0 : {{table_dtype}} , %b2 : complex<{{input_dtype}}>, %b1 : {{dtype}}):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = linalg.index 3 : index
      %mod = arith.remui %3, %c2 : index
      %cos = math.cos %b0 : {{dtype}}
      %sin = math.sin %b0 : {{dtype}}
      %comp_rot = complex.create %cos, %sin : complex<{{dtype}}>
      {% if input_dtype != dtype %}
        %in_real = complex.re %b2: complex<{{input_dtype}}>
        %in_imag = complex.im %b2: complex<{{input_dtype}}>
        %real_extf = arith.extf %in_real
          : {{input_dtype}} to {{dtype}}
        %imag_extf = arith.extf %in_imag
          : {{input_dtype}} to {{dtype}}
        %4 = complex.create %real_extf, %imag_extf : complex<{{dtype}}>
        %result = complex.mul %4, %comp_rot : complex<{{dtype}}>
      {% else %}
        %result = complex.mul %b2, %comp_rot : complex<{{dtype}}>
      {% endif %}
      %real = complex.re %result : complex<{{dtype}}>
      %imag = complex.im %result : complex<{{dtype}}>
      %cmp = arith.cmpi eq, %mod, %c0 : index
      %val = arith.select %cmp, %real, %imag : {{dtype}}
      linalg.yield %val : {{dtype}}
  } -> !output_tensor_type

  util.return %result : !output_tensor_type
}

}
