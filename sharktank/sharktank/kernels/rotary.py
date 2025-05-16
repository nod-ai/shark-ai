# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.support.conversions import (
    IREE_TYPE_ASM_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_IREE_TYPE_ASM,
)
from typing import Optional, cast
from iree.compiler.ir import Type
from sharktank.kernels.base import *
from iree.turbine.runtime.op_reg import AttrArg

__all__ = [
    "apply_rotary_embedding",
]


def apply_rotary_embedding(
    input: torch.Tensor, table: torch.Tensor, /, *, dtype: Optional[torch.dtype] = None
):
    if dtype is None:
        dtype = input.dtype
    assert dtype == table.dtype
    return _apply_rotary_embedding(
        input=input, table=table, dtype=TORCH_DTYPE_TO_IREE_TYPE_ASM[dtype]
    )


@CustomOp.register(library=LIBRARY)
class _apply_rotary_embedding(CustomOp):

    signature = (
        "apply_rotary_embedding(Tensor input, Tensor table, str dtype) -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        inputs_desc = ksel.arg_tensor(0)
        table_desc = ksel.arg_tensor(1)
        dtype_desc = ksel.attr_str(2)
        dtype = IREE_TYPE_ASM_TO_TORCH_DTYPE[dtype_desc.v]
        out_desc = ksel.return_new_tensor(inputs_desc.t.shape, dtype=dtype)
        specialize_all_known_dims(inputs_desc)
        specialize_all_known_dims(table_desc)
        specialize_all_known_dims(out_desc)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):

        input = kb.arg_value(0)
        table = kb.arg_value(1)
        dtype_str = cast(AttrArg, ksel.arg_descs[2]).v
        dtype = Type.parse(dtype_str)

        input_tensor_type = RankedTensorType(input.type)
        table_tensor_type = RankedTensorType(table.type)
        output_tensor_type = RankedTensorType.get(input_tensor_type.shape, dtype)

        input_asm_type, input_ident, input_dtype = unpack_tensor_type(input.type)
        table_asm_type, table_ident, table_dtype = unpack_tensor_type(table.type)

        # Generate specialization signature and types.
        bs = input.type.shape[0]
        sl = input.type.shape[1]
        sl = "D" if sl < 0 else sl
        heads = input.type.shape[2]
        dims = input.type.shape[3]

        template_file = "rotary_embedding.mlir"
        target_function_name = f"sharktank_rotary_embedding_{bs}_{sl}_{heads}_{dims}_{input_dtype}_{table_dtype}_{dtype}"

        # Template params.
        input_tensor_type = input_asm_type
        table_tensor_type = table_asm_type

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            func_name=target_function_name,
            input_tensor_type=input_tensor_type,
            table_tensor_type=table_tensor_type,
            output_tensor_type=str(output_tensor_type),
            bs=bs,
            sl=sl,
            heads=heads,
            dims=dims,
            input_dtype=str(input_dtype),
            table_dtype=str(table_dtype),
            dtype=str(dtype),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
