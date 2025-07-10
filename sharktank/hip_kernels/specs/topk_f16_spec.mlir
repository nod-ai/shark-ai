// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "{{HIP_ARCH}}", ukernels = "none"}>

module attributes {transform.with_named_sequence} {
  util.func private @topk_3d_f16_entry_point(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xi32>) -> (tensor<?x8xf16>, tensor<?x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf16>

    %dim2 = tensor.dim %arg1, %c0 : tensor<?x?xi32>
    %dim3 = tensor.dim %arg1, %c1 : tensor<?x?xi32>

    %dim1_i32 = arith.index_cast %dim1 : index to i32
    %4:2 = hal.dispatch.extern "topk_F16I32"[%dim0](%dim1_i32, %arg0, %arg1) : (i32, tensor<?x?xf16>{%dim0, %dim1}, tensor<?x?xi32>{%dim2, %dim3}) -> tensor<?x8xf16>{%dim0}, tensor<?x8xi32>{%dim0}
      count(%device: !hal.device, %batchSize: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        hal.return %batchSize, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<constants = 1, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "{{BUILD}}/compiled_kernels/topk_fp16_ukernel.c.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [64 : index, 1 : index, 1 : index]}
    util.return %4#0, %4#1 : tensor<?x8xf16>, tensor<?x8xi32>
  }

  transform.named_sequence @match_topk(%linalg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %linalg ["iree_linalg_ext.topk"] : !transform.any_op
    %in0 = transform.get_operand %linalg[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[1], 64 : !transform.any_value
    %in1 = transform.get_operand %linalg[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in1 = tensor<?x?xi32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in1[1], 64 : !transform.any_value
    %out0 = transform.get_operand %linalg[2] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out0 = tensor<?x8xf16> : !transform.any_value
    %out1 = transform.get_operand %linalg[3] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out1 = tensor<?x8xi32> : !transform.any_value
    transform.yield %linalg : !transform.any_op
  }

  transform.named_sequence @cast_and_call_topk(%topk: !transform.any_op {transform.readonly}) {
    %module = transform.util.get_nearest_symbol_table %topk : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @topk_3d_f16_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %topk[0,1] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %topk[all] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %topk {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_topk -> @cast_and_call_topk
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
