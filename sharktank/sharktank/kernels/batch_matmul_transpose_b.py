# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

from iree.compiler.ir import IntegerType, FloatType

__all__ = [
    "batch_matmul_transpose_b",
]


@CustomOp.register(library=LIBRARY)
class batch_matmul_transpose_b(CustomOp):
    """Generic block scaled matmul with transposed RHS.

    The LHS is expected to be a 3d tensor of shape [B, M, K]. RHS must be
    [B, N, K].

    The kernel will be specialized for all values of N, K and LHS dtype.
    """

    signature = "batch_matmul_transpose_b(Tensor lhs, Tensor rhs) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        lhs_desc = ksel.arg_tensor(0)  # Shape [B, M, K]
        rhs_desc = ksel.arg_tensor(1)  # Shape [B, N, K]

        # Rank check.
        torch._check(
            len(lhs_desc.t.shape) == 3,
            lambda: f"batch_matmul_transpose_b arg 'lhs': Expected 3d tensor (got {lhs_desc.t.shape})",
        )

        # Rank check.
        torch._check(
            len(rhs_desc.t.shape) == 3,
            lambda: f"batch_matmul_transpose_b arg 'rhs': Expected 3d tensor (got {rhs_desc.t.shape})",
        )

        # a arg
        lhs_batch, lhs_m, lhs_k = lhs_desc.t.shape

        # d arg
        rhs_batch, rhs_n, rhs_k = rhs_desc.t.shape
        torch._check(
            rhs_k == lhs_k,
            lambda: f"batch_matmul_transpose_b arg 'rhs': Incorrect shape (got {rhs_desc.t.shape})",
        )

        # Batch must be pre-broadcast.
        torch._check(
            lhs_batch == rhs_batch,
            lambda: f"batch_matmul_transpose_b: Batch dims must match ({lhs_desc.t.shape} vs {rhs_desc.t.shape})",
        )
        # Shape batch, m, n
        c_desc = ksel.return_new_tensor([lhs_batch, lhs_m, rhs_n], dtype=torch.float32)
        specialize_all_known_dims(lhs_desc)
        specialize_all_known_dims(rhs_desc)
        specialize_all_known_dims(c_desc)

        # Require specialize on K, N.
        lhs_desc.specialize_dims(-1)
        rhs_desc.specialize_dims(-1, -2)
        c_desc.specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        lhs = kb.arg_value(0)
        rhs = kb.arg_value(1)
        result_desc = ksel.result_descs[0]

        # Generate specialization signature and types.
        a_asm_type, a_ident, _ = unpack_tensor_type(lhs.type)
        b_asm_type, b_ident, _ = unpack_tensor_type(rhs.type)
        accum_type = FloatType.parse("f32")
        spec_sig = f"L{a_ident}_R{b_ident}"
        template_file = "batch_matmul_transpose_b.mlir"
        target_function_name = f"sharktank_batch_matmul_transpose_b_{spec_sig}"
        cst_zero = "0" if IntegerType.isinstance(accum_type) else "0."
        # Template params.
        c_asm_type = f"tensor<{'x'.join('?' if d is None else str(d) for d in result_desc.spec_dims)}x{accum_type}>"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            spec_sig=spec_sig,
            a_asm_type=a_asm_type,
            b_asm_type=b_asm_type,
            c_asm_type=c_asm_type,
            dtype=str(accum_type),
            cst_zero=cst_zero,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
