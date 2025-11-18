# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *
import torch

__all__ = [
    "iree_topk",
]


@CustomOp.register(library=LIBRARY)
class iree_topk(CustomOp):

    signature = "iree_topk(Tensor input, Tensor indices, int k) -> (Tensor values, Tensor indices)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        indices_desc = ksel.arg_tensor(1)
        k = ksel.attr_int(2).v
        values_desc = ksel.return_new_tensor(
            [inputs_desc.t.shape[0], k],
            dtype=inputs_desc.t.dtype,
        )
        indices_desc = ksel.return_new_tensor(
            [inputs_desc.t.shape[0], k], dtype=torch.int32
        )

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        input = kb.arg_value(0)
        indices = kb.arg_value(1)

        result_desc = ksel.result_descs[0]
        result_shape = result_desc.t.shape
        bs = result_shape[0]
        k = result_shape[-1]

        bs = "D" if bs == -1 else bs

        input_tensor_type = RankedTensorType(input.type)
        input_asm_type, input_ident, input_dtype = unpack_tensor_type(input.type)

        # Generate specialization signature and types.
        template_file = "topk_dynamic.mlir"
        target_function_name = f"amdsharktank_topk_{bs}_{k}_{input_dtype}"

        # Template params
        input_tensor_type = input_asm_type
        indices_tensor_type = f"tensor<?x?xi32>"

        values_out_tensor_type = ksel.result_descs[0].mlir_type_asm
        indices_out_tensor_type = ksel.result_descs[1].mlir_type_asm

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            input_tensor_type=input_tensor_type,
            indices_tensor_type=indices_tensor_type,
            values_out_tensor_type=values_out_tensor_type,
            indices_out_tensor_type=indices_out_tensor_type,
            bs=bs,
            k=k,
            dtype=str(input_dtype),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
