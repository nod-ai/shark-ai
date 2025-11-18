# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *

__all__ = [
    "apply_rotary_embedding",
]


@CustomOp.register(library=LIBRARY)
class apply_rotary_embedding(CustomOp):

    signature = "apply_rotary_embedding(Tensor input, Tensor table) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        table_desc = ksel.arg_tensor(1)
        out_desc = ksel.return_new_tensor(
            inputs_desc.t.shape, dtype=inputs_desc.t.dtype
        )
        specialize_all_known_dims(inputs_desc)
        specialize_all_known_dims(table_desc)
        specialize_all_known_dims(out_desc)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):

        input = kb.arg_value(0)
        table = kb.arg_value(1)

        input_tensor_type = RankedTensorType(input.type)
        table_tensor_type = RankedTensorType(table.type)

        input_asm_type, input_ident, input_dtype = unpack_tensor_type(input.type)
        table_asm_type, table_ident, table_dtype = unpack_tensor_type(table.type)

        assert input_dtype == table_dtype

        # Generate specialization signature and types.
        bs = input.type.shape[0]
        sl = input.type.shape[1]
        sl = "D" if sl < 0 else sl
        heads = input.type.shape[2]
        dims = input.type.shape[3]

        template_file = "rotary_embedding.mlir"
        target_function_name = (
            f"amdsharktank_rotary_embedding_{bs}_{sl}_{heads}_{dims}_{input_dtype}"
        )

        # Template params.
        input_tensor_type = input_asm_type
        table_tensor_type = table_asm_type

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            input_tensor_type=input_tensor_type,
            table_tensor_type=table_tensor_type,
            bs=bs,
            sl=sl,
            heads=heads,
            dims=dims,
            dtype=str(input_dtype),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
