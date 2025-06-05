# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
import torch

__all__ = [
    "iree_topk",
]


@CustomOp.register(library=LIBRARY)
class iree_topk(CustomOp):

    signature = "iree_topk(Tensor input, int k) -> (Tensor values, Tensor indices)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        k = ksel.attr_int(1).v
        values_desc = ksel.return_new_tensor(
            [inputs_desc.t.shape[0], inputs_desc.t.shape[1], k],
            dtype=inputs_desc.t.dtype,
        )
        indices_desc = ksel.return_new_tensor(
            [inputs_desc.t.shape[0], inputs_desc.t.shape[1], k], dtype=torch.int32
        )
        # specialize_all_known_dims(inputs_desc)
        # specialize_all_known_dims(values_desc)
        # specialize_all_known_dims(indices_desc)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        input = kb.arg_value(0)

        result_desc = ksel.result_descs[0]
        result_shape = result_desc.t.shape
        k = result_shape[-1]

        input_tensor_type = RankedTensorType(input.type)
        input_asm_type, input_ident, input_dtype = unpack_tensor_type(input.type)

        # Generate specialization signature and types.
        bs = input.type.shape[0]
        sl = input.type.shape[1]
        sl = "?" if sl < 0 else sl  # Use ? for dynamic dimensions in MLIR

        template_file = "topk_dynamic.mlir"
        target_function_name = f"sharktank_topk_{k}_{input_dtype}"

        # Template params
        input_tensor_type = input_asm_type
        indices_tensor_type = f"tensor<?x?x?xi32>"
        values_tensor_type = f"tensor<?x?x{k}x{input_dtype}>"
        indices_out_tensor_type = f"tensor<?x?x{k}xi32>"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            input_tensor_type=input_tensor_type,
            indices_tensor_type=indices_tensor_type,
            values_tensor_type=values_tensor_type,
            indices_out_tensor_type=indices_out_tensor_type,
            bs=bs,
            sl=sl,
            k=k,
            dtype=str(input_dtype),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
