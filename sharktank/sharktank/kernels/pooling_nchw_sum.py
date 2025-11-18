# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *

import math

import torch

__all__ = [
    "pooling_nchw_sum",
]


@CustomOp.register(library=LIBRARY)
class pooling_nchw_sum(CustomOp):
    """Generic pooling sum operates on explicitly padded inputs."""

    signature = "pooling_nchw_sum(Tensor inputs, int[] weights_size, int[] strides, int[] dilations) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        kernel_size_desc = ksel.attr_list_int(1)  # Shape [2]
        strides_desc = ksel.attr_list_int(2)  # Shape [2]
        dilations_desc = ksel.attr_list_int(3)  # Shape [2]

        # unpack
        n, c, h_pad, w_pad = inputs_desc.t.shape

        kernel_size = kernel_size_desc.v
        strides = strides_desc.v
        dilations = dilations_desc.v

        # pooling shape math
        h_out = math.floor((h_pad - kernel_size[0]) / strides[0] + 1)
        w_out = math.floor((w_pad - kernel_size[1]) / strides[1] + 1)
        c_desc = ksel.return_new_tensor([n, c, h_out, w_out], dtype=inputs_desc.t.dtype)
        specialize_all_known_dims(inputs_desc)
        specialize_all_known_dims(c_desc)

        # Require specialization of width/height.
        inputs_desc.specialize_dims(-1, -2)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        inputs = kb.arg_value(0)
        inputs_tensor_type = RankedTensorType(inputs.type)
        kernel_size = ksel.arg_descs[1].v
        strides = ksel.arg_descs[2].v
        dilations = ksel.arg_descs[3].v
        result_desc = ksel.result_descs[0]
        result_shape = result_desc.t.shape
        _, _, H_out, W_out = result_desc.t.shape

        # Generate specialization signature and types.
        inputs_asm_type, inputs_ident, accum_type = unpack_tensor_type(inputs.type)
        spec_sig = (
            f"I{inputs_ident}"
            f"_K{kernel_size[0]}x{kernel_size[1]}"
            f"_S{strides[0]}x{strides[1]}"
            f"_D{dilations[0]}x{dilations[1]}"
        )
        template_file = "pooling_nchw_sum.mlir"
        target_function_name = f"amdsharktank_pooling_nchw_sum_{spec_sig}"

        # Template params.
        result_asm_type = f"tensor<{'x'.join('?' if d is None else str(d) for d in result_desc.spec_dims)}x{accum_type}>"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            spec_sig=spec_sig,
            inputs_asm_type=inputs_asm_type,
            result_asm_type=result_asm_type,
            H_out=str(H_out),
            W_out=str(W_out),
            ks_H=str(kernel_size[0]),
            ks_W=str(kernel_size[1]),
            strides_H=str(strides[0]),
            strides_W=str(strides[1]),
            dilations_H=str(dilations[0]),
            dilations_W=str(dilations[1]),
            accum_type=accum_type,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
