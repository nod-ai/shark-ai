# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *

import torch

__all__ = [
    "flash_attention",
    "masked_flash_attention",
]

BATCH = DynDim.BATCH
NUM_HEADS = DynDim.NUM_HEADS
M = DynDim.M
K1 = StaticDim.K1
K2 = DynDim.K2
N = StaticDim.N

<<<<<<< HEAD
I_DTYPE = Dtype.I_DTYPE
M_DTYPE = Dtype.M_DTYPE
S_DTYPE = Dtype.S_DTYPE
O_DTYPE = Dtype.O_DTYPE(torch.float32)
=======
@CustomOp.register(library=LIBRARY)
class masked_flash_attention(CustomOp):

    signature = "masked_flash_attention(Tensor q, Tensor k, Tensor v, Tensor? a, Tensor scale) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        q_desc = ksel.arg_tensor(0)  # Shape b, l, d
        k_desc = ksel.arg_tensor(1)  # Shape b, s, d
        v_desc = ksel.arg_tensor(2)  # Shape b, s, e
        a_desc = ksel.arg_tensor(3)  # Shape b, l, s
        s_desc = ksel.arg_tensor(4)

        q_bs = q_desc.t.shape[:-2]
        k_bs = k_desc.t.shape[:-2]
        v_bs = v_desc.t.shape[:-2]
        a_bs = a_desc.t.shape[:-2]

        bs = len(q_bs)

        # Note: kernel does collapse dims to get to a single batch/head dim
        torch._check(len(q_bs) == 2, lambda: f"TODO: batch dims {bs} not supported")

        q_l, q_d = q_desc.t.shape[-2:]
        k_s, k_d = k_desc.t.shape[-2:]
        v_s, v_e = v_desc.t.shape[-2:]

        torch._check(
            q_desc.t.dtype.is_floating_point
            and k_desc.t.dtype.is_floating_point
            and v_desc.t.dtype.is_floating_point
            and s_desc.t.dtype.is_floating_point,
            lambda: f"masked_flash_attention: Expected floating point",
        )
        torch._check(
            q_desc.t.dtype == k_desc.t.dtype == v_desc.t.dtype,
            lambda: f"masked_flash_attention: Expected matching dtypes.",
        )

        for q_b, k_b, v_b in zip(q_bs, k_bs, v_bs):
            torch._check(
                q_b == k_b and q_b == v_b,
                lambda: f"expected matching batch dims: {q_b}, {k_b}, {v_b}",
            )

        torch._check(q_d == k_d, lambda: f"expected matching qk features: {q_d}, {k_d}")

        torch._check(k_s == v_s, lambda: f"expected matching kv length: {q_d}, {k_d}")

        q_desc.specialize_dims(0, 1, -1)
        k_desc.specialize_dims(0, 1, -1)
        v_desc.specialize_dims(0, 1, -1)

        # Result 0: Shape batch..., m, n
        ksel.return_new_tensor((*q_bs, q_l, v_e), dtype=torch.float32).specialize_dims(
            0, 1, -1
        )

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        q = kb.arg_value(0)
        k = kb.arg_value(1)
        v = kb.arg_value(2)
        a = kb.arg_value(3)
        scale = kb.arg_value(4)

        q_tensor_type = RankedTensorType(q.type)
        scale_tensor_type = RankedTensorType(scale.type)
        v_tensor_type = RankedTensorType(v.type)

        b1, b2, l, d = q_tensor_type.shape
        _, _, s, e = v_tensor_type.shape

        # Unspecialized dims will be negative
        l = l if l >= 0 else "?"
        s = s if s >= 0 else "?"
        b = str(int(b1) * int(b2))
        i_type_str = str(q_tensor_type.element_type)
        scale_type_str = str(scale_tensor_type.element_type)
        a_type_str = str(RankedTensorType(a.type).element_type)
        # TODO: enable f16 output type via arg
        o_type_str = "f32"

        target_function_name = f"sharktank_masked_flash_attention_{b1}_{b2}_{d}_{e}_{i_type_str}_{a_type_str}_{scale_type_str}_{o_type_str}"
        kwargs = {
            "b": b,
            "b1": b1,
            "b2": b2,
            "l": l,
            "d": d,
            "s": s,
            "e": e,
            "a_dtype": a_type_str,
            "i_dtype": i_type_str,
            "scale_dtype": scale_type_str,
            "o_dtype": o_type_str,
            "func_name": target_function_name,
        }
        template_file = "masked_flash_attention.mlir"
        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(*call_function(target_function, q, k, v, scale, a))
        pass
>>>>>>> 6100a942 (WIP)


@mlir_kernel(
    inputs=(
        MLIRTensor[BATCH, NUM_HEADS, M, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, N, I_DTYPE],
        MLIRTensor[S_DTYPE],
    ),
    results=(MLIRTensor[BATCH, NUM_HEADS, M, N, O_DTYPE],),
)
def flash_attention(q, k, v, scale, result=None):
    mlir = """
    module {
    util.func @{{kernel_name}}(%q : !q, %k : !k, %v: !v, %scale: !scale) -> !result {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %batch = tensor.dim %q, %c0 : !q
      %num_heads = tensor.dim %q, %c1 : !q
      %m = tensor.dim %q, %c2 : !q

      %empty = tensor.empty(%batch, %num_heads, %m) : !result

      %s_c = tensor.extract %scale[] : !scale

      %result = iree_linalg_ext.attention {
        indexing_maps = [
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, N)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> ()>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, N)>
        ]
      }
      ins(%q, %k, %v, %s_c : !q, !k, !v, !scale_dtype)
      outs(%empty : !result) {
        ^bb0(%score : f32):
          iree_linalg_ext.yield %score : f32
      } -> !result

      util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)


@mlir_kernel(
    inputs=(
        MLIRTensor[BATCH, NUM_HEADS, M, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, K1, I_DTYPE],
        MLIRTensor[BATCH, NUM_HEADS, K2, N, I_DTYPE],
        MLIRTensor[M, K2, M_DTYPE],
        MLIRTensor[S_DTYPE],
    ),
    results=(MLIRTensor[BATCH, NUM_HEADS, M, N, O_DTYPE],),
)
def masked_flash_attention(q, k, v, mask, scale, result=None):
    mlir = """
    module {
    util.func @{{kernel_name}}(%q : !q, %k : !k, %v: !v, %mask : !mask, %scale: !scale) -> !result {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index

      %batch = tensor.dim %q, %c0 : !q
      %num_heads = tensor.dim %q, %c1 : !q
      %m = tensor.dim %q, %c2 : !q

      %empty = tensor.empty(%batch, %num_heads, %m) : !result

      %s_c = tensor.extract %scale[] : !scale

      %result = iree_linalg_ext.attention {
        indexing_maps = [
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, K1)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, K2, N)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> ()>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (M, K2)>,
          affine_map<(BATCH, NUM_HEADS, M, N, K1, K2) -> (BATCH, NUM_HEADS, M, N)>
        ]
      }
      ins(%q, %k, %v, %s_c, %mask : !q, !k, !v, !scale_dtype, !mask)
      outs(%empty : !result) {
        ^bb0(%score : f32):
          iree_linalg_ext.yield %score : f32
      } -> !result

      util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)
