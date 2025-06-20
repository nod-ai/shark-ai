# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from .base import *

__all__ = [
    "batched_block_scaled_mmt_fp4",
]


@CustomOp.register(library=LIBRARY)
class batched_block_scaled_mmt_fp4(CustomOp):
    """Batched block scaled matrix multiplication for FP4 quantized weights.

    This kernel operates on PlanarQuantizedTensor with BlockScaledFp4Layout:

    * `d`: `[N, K // BLOCK_SIZE]` (per-block scales)
    * `qs`: `[N, K // BLOCK_SIZE, BLOCK_SIZE]` (unpacked FP4 indices as uint8, values 0-15)

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of N, K, block size, and LHS dtype.
    """

    signature = (
        "batched_block_scaled_mmt_fp4(Tensor a, Tensor d, Tensor qs) -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape [B, M, K]
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE]
        qs_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE]

        # a arg
        *a_batch_dims, a_m, a_k = a_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"batched_block_scaled_mmt_fp4 arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )
        torch._check(
            len(a_batch_dims) == 1,
            lambda: f"batched_block_scaled_mmt_fp4 arg 'a': Expected 3d tensor with shape [B, M, K], got {len(a_desc.t.shape)}d tensor {a_desc.t.shape}",
        )

        # qs arg - unpacked FP4 indices
        *qs_batch_dims, qs_n, qs_group0, qs_bs = qs_desc.t.shape
        torch._check(
            (
                len(qs_batch_dims) == 0
                or (len(qs_batch_dims) == 1 and qs_batch_dims == a_batch_dims)
            )
            and (qs_group0 * qs_bs) == a_k,
            lambda: f"batched_block_scaled_mmt_fp4 arg 'qs': Expected shape [N, K//BS, BS] or [B, N, K//BS, BS] with K//BS * BS = {a_k}, got {qs_desc.t.shape} (computed K = {qs_group0 * qs_bs})",
        )
        torch._check(
            qs_desc.t.dtype == torch.uint8,
            lambda: f"batched_block_scaled_mmt_fp4 arg 'qs': Expected uint8 (got {qs_desc.t.dtype})",
        )
        block_size = qs_bs

        # d arg - per-block scales
        *d_batch_dims, d_n, d_group0 = d_desc.t.shape
        torch._check(
            (
                d_batch_dims == qs_batch_dims
                and (d_group0 * block_size) == a_k
                and d_n == qs_n
            ),
            lambda: f"batched_block_scaled_mmt_fp4 arg 'd': Expected shape [N, K//BS] or [B, N, K//BS] with N={qs_n}, K//BS={a_k // block_size}, got {d_desc.t.shape} (batch_dims mismatch: {d_batch_dims} vs {qs_batch_dims}, K mismatch: {d_group0 * block_size} vs {a_k}, N mismatch: {d_n} vs {qs_n})",
        )
        torch._check(
            d_desc.t.dtype.is_floating_point,
            lambda: f"batched_block_scaled_mmt_fp4 arg 'd': Expected floating point (got {d_desc.t.dtype})",
        )

        # Specialize on K, N, BS
        a_desc.specialize_dims(-1)
        if len(qs_batch_dims) == 0:
            qs_desc.specialize_all_dims()
            d_desc.specialize_all_dims()
        else:
            qs_desc.specialize_dims(1, 2, 3)
            d_desc.specialize_dims(1, 2)

        # Shape [B, M, N]
        c_desc = ksel.return_new_tensor(a_batch_dims + [a_m, d_n], dtype=a_desc.t.dtype)
        c_desc.specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(2)
        qs_tensor_type = RankedTensorType(qs.type)

        rank = a_tensor_type.rank
        k = a_tensor_type.get_dim_size(rank - 1)
        *qs_batch_dims, n, group0, bs = qs_tensor_type.shape
        batched_rhs = len(qs_batch_dims) == 1
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "mmt_block_scaled_fp4.mlir"
        target_function_name = (
            f"sharktank_batched_block_scaled_mmt_fp4_3d_{n}_{k}_{bs}_"
            f"{a_type_str}_{batched_rhs}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            bs=bs,
            group0=group0,
            a_type=a_type_str,
            scale_type=scale_type_str,
            batched_rhs=batched_rhs,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
