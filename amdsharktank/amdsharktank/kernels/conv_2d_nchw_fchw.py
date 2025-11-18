# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *

import math

import torch

__all__ = [
    "conv_2d_nchw_fchw",
]


_LEGAL_OUTPUT_TYPES = {
    # Legal integer types.
    (torch.int8, torch.int8, "torch.int8"): torch.int8,
    (torch.int8, torch.int8, "torch.int16"): torch.int16,
    (torch.int8, torch.int8, "torch.int32"): torch.int32,
    (torch.int16, torch.int16, "torch.int16"): torch.int16,
    (torch.int16, torch.int16, "torch.int32"): torch.int32,
    # Legal fp types.
    (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, "torch.float16"): torch.float16,
    (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, "torch.float32"): torch.float32,
    (torch.float16, torch.float16, "torch.float16"): torch.float16,
    (torch.float16, torch.float16, "torch.float32"): torch.float32,
    (torch.float32, torch.float32, "torch.float32"): torch.float32,
}

# Ensure that every output type above is in this table.
_OUTPUT_TYPE_TO_ASM = {
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.float8_e4m3fnuz: "f8E4M3FNUZ",
    torch.float16: "f16",
    torch.float32: "f32",
}


@CustomOp.register(library=LIBRARY)
class conv_2d_nchw_fchw(CustomOp):
    """Generic convolution on explicitly padded inputs on mixed precision
    operands. This is meant to be used for pure integer convolutions and
    supports automatic type promotion. It can be invoked either with uniform
    types or any combination of signed integer inputs and a wider, signed
    output type. Note that the ops this lower to only support signed extension.

    Will be specialized for all values of strides, padding, dilations, and LHS dtype.
    """

    signature = "conv_2d_nchw_fchw(Tensor inputs, Tensor weights, Tensor bias, int[] strides, int[] dilations, str output_dtype) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        weights_desc = ksel.arg_tensor(1)
        bias_desc = ksel.arg_tensor(2)
        strides_desc = ksel.attr_list_int(3)  # Shape [2]
        dilations_desc = ksel.attr_list_int(4)  # Shape [2]
        output_dtype_name = ksel.attr_str(5).v
        if "." not in output_dtype_name:
            output_dtype_name = f"torch.{output_dtype_name}"

        # unpack
        n, c, h_pad, w_pad = inputs_desc.t.shape
        f, g, k0, k1 = weights_desc.t.shape
        (b,) = bias_desc.t.shape

        strides = strides_desc.v
        dilations = dilations_desc.v

        # validate types
        type_tuple = (inputs_desc.t.dtype, weights_desc.t.dtype, output_dtype_name)
        output_dtype = _LEGAL_OUTPUT_TYPES.get(type_tuple)
        torch._check(
            output_dtype is not None,
            lambda: f"conv_2d_nchw_fchw unsupported dtype combination: got {type_tuple}. allowed:\n  {list(_LEGAL_OUTPUT_TYPES.keys())}",
        )
        torch._check(
            bias_desc.t.dtype == output_dtype,
            lambda: f"conv_2d_nchw_fchw bias type must match output dtype: got {bias_desc.t.dtype}, expected {output_dtype}",
        )

        # check
        torch._check(
            b == f,
            lambda: f"conv_2d_nchw_fchw bias shape should match out_channels shape but got {b} instead of {f}",
        )
        torch._check(
            len(strides) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 strides; strides: {strides}",
        )
        torch._check(
            len(dilations) == 2,
            lambda: f"conv_2d_nchw_fchw requires exactly 2 dilations; dilations: {dilations}",
        )

        # convolution shape math
        h_out = math.floor((h_pad - dilations[0] * (k0 - 1) - 1) / strides[0] + 1)
        w_out = math.floor((w_pad - dilations[1] * (k1 - 1) - 1) / strides[1] + 1)
        c_desc = ksel.return_new_tensor([n, f, h_out, w_out], dtype=output_dtype)
        specialize_all_known_dims(inputs_desc)
        specialize_all_known_dims(weights_desc)
        specialize_all_known_dims(bias_desc)
        specialize_all_known_dims(c_desc)

        # Always specialize the our W/H as we presently do not materialize the output
        # size computation in the IR.
        inputs_desc.specialize_dims(-1, -2)
        weights_desc.specialize_dims(-1, -2)
        c_desc.specialize_dims(-1, -2)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        inputs = kb.arg_value(0)
        strides = ksel.arg_descs[3].v
        dilations = ksel.arg_descs[4].v
        result_desc = ksel.result_descs[0]
        result_shape = result_desc.t.shape
        H_out = result_shape[2]
        W_out = result_shape[3]

        # Generate specialization signature and types.
        result_dtype = result_desc.t.dtype
        accum_type = _OUTPUT_TYPE_TO_ASM[result_dtype]
        inputs_asm_type, inputs_ident, _ = unpack_tensor_type(inputs.type)
        weights_asm_type, weights_ident, _ = unpack_tensor_type(kb.arg_value(1).type)
        bias_asm_type, bias_ident, _ = unpack_tensor_type(kb.arg_value(2).type)
        spec_sig = (
            f"I{inputs_ident}_W{weights_ident}_B{bias_ident}"
            f"_O{accum_type}"
            f"_S{strides[0]}x{strides[1]}"
            f"_D{dilations[0]}x{dilations[1]}"
        )
        template_file = "conv_2d_nchw_fchw.mlir"
        target_function_name = f"amdsharktank_conv_2d_nchw_fchw_{spec_sig}"

        # Template params.
        result_asm_type = f"tensor<{'x'.join('?' if d is None else str(d) for d in result_desc.spec_dims)}x{accum_type}>"
        strides = [str(i) for i in strides]
        dilations = [str(i) for i in dilations]
        H_out = str(H_out)
        W_out = str(W_out)

        is_floating_point = result_dtype.is_floating_point
        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            spec_sig=spec_sig,
            inputs_asm_type=inputs_asm_type,
            weights_asm_type=weights_asm_type,
            bias_asm_type=bias_asm_type,
            result_asm_type=result_asm_type,
            H_out=H_out,
            W_out=W_out,
            strides_H=strides[0],
            strides_W=strides[1],
            dilations_H=dilations[0],
            dilations_W=dilations[1],
            accum_type=str(accum_type),
            zero=("0." if is_floating_point else "0"),
            add_op=("arith.addf" if is_floating_point else "arith.addi"),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
