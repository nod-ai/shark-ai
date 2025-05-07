# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "elementwise_tensor_tensor",
    "elementwise_tensor",
]

@CustomOp.register(library=LIBRARY)
class elementwise_tensor(CustomOp):
    """TODO"""
    signature = "elementwise_tensor(Tensor x, str op) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        x_desc = ksel.arg_tensor(0)
        op_str = ksel.attr_str(1).v

        ksel.return_new_tensor(x_desc.t.shape, dtype=x_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        x = kb.arg_value(0)
        x_tensor_type = RankedTensorType(x.type)
        op_str = ksel.arg_descs[1].v

        rank = x_tensor_type.rank
        element_type = x_tensor_type.element_type

        template_file = "elementwise_tensor.mlir"
        target_function_name = (
            f"sharktank_elementwise_tensor_{op_str}_{rank}_{element_type}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            op_str=op_str,
            rank=rank,
            dtype=element_type,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))

@CustomOp.register(library=LIBRARY)
class elementwise_tensor_tensor(CustomOp):
    """TODO"""

    signature = "elementwise_tensor_tensor(Tensor x, Tensor y, str op) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        x_desc = ksel.arg_tensor(0)
        y_desc = ksel.arg_tensor(1)
        ret_shape = []
        op_str = ksel.attr_str(2).v
        x_spec = []
        y_spec = []

        torch._check(
            len(x_desc.t.shape) == len(y_desc.t.shape),
            lambda: f"elementwise_tensor_tensor: Inputs must be the same rank. {x_desc.t.shape} != {y_desc.t.shape}",
        )
        # Shape match
        for i in range(len(x_desc.t.shape)):
            # Broadcast
            ret_shape.append(max(x_desc.t.shape[i], y_desc.t.shape[i]))
            if x_desc.t.shape[i] == 1 or y_desc.t.shape[i] == 1:
                if x_desc.t.shape[i] == 1:
                    x_spec.append(i)
                if y_desc.t.shape[i] == 1:
                    y_spec.append(i)
                continue
            torch._check(
                x_desc.t.shape[i] == y_desc.t.shape[i],
                lambda: f"elementwise_tensor_tensor: Incompatible shapes. {x_desc.t.shape} != {y_desc.t.shape} at position {i}",
            )
        torch._check(
            x_desc.t.dtype == y_desc.t.dtype,
            lambda: f"elementwise_tensor_tensor: Types don't match. {x_desc.t.dtype} != {y_desc.t.dtype}",
        )

        x_desc.specialize_dims(*x_spec)
        y_desc.specialize_dims(*y_spec)
        ksel.return_new_tensor(ret_shape, dtype=x_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        x = kb.arg_value(0)
        x_tensor_type = RankedTensorType(x.type)
        y = kb.arg_value(1)
        y_tensor_type = RankedTensorType(y.type)
        op_str = ksel.arg_descs[2].v

        rank = x_tensor_type.rank
        element_type = x_tensor_type.element_type

        x_ones = []
        y_ones = []
        xshape = []
        yshape = []
        x_reassoc = []
        x_accum = []
        y_reassoc = []
        y_accum = []

        def update(lst, elem, accum, b):
            if not b:
                lst.append([*accum, elem])
                accum.clear()
            else:
                if lst != []:
                    lst[-1].append(*accum, elem)
                    accum.clear()
                else:
                    accum.append(elem)

        for i in range(rank):
            if x_tensor_type.shape[i] == 1:
                x_ones.append(f"{i}")
                xshape.append("1")
            else:
                xshape.append("?")
            if y_tensor_type.shape[i] == 1:
                y_ones.append(f"{i}")
                yshape.append("1")
            else:
                yshape.append("?")
            update(x_reassoc, i, x_accum, x_tensor_type.shape[i] == 1)
            update(y_reassoc, i, y_accum, y_tensor_type.shape[i] == 1)
        x_broadcast = ", ".join(x_ones)
        y_broadcast = ", ".join(y_ones)
        broadcast_str = "x".join(x_ones) + "_" + "x".join(y_ones)

        template_file = "elementwise_tensor_tensor.mlir"
        target_function_name = f"sharktank_elementwise_tensor_tensor_{op_str}_{rank}_{element_type}_{broadcast_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            op_str=op_str,
            rank=rank,
            dtype=element_type,
            x_broadcast=x_broadcast,
            y_broadcast=y_broadcast,
            x_size=len(x_ones),
            y_size=len(y_ones),
            xshape="x".join(xshape),
            yshape="x".join(yshape),
            broadcast_str=broadcast_str,
            x_reassociation=str(x_reassoc),
            y_reassociation=str(y_reassoc),
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
