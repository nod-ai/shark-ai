# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *

import torch

__all__ = [
    "flash_attention",
]


@CustomOp.register(library=LIBRARY)
class flash_attention(CustomOp):

    signature = "flash_attention(Tensor q, Tensor k, Tensor v, Tensor? a, Tensor scale) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        q_desc = ksel.arg_tensor(0)  # Shape b, l, d
        k_desc = ksel.arg_tensor(1)  # Shape b, s, d
        v_desc = ksel.arg_tensor(2)  # Shape b, s, e
        a_desc = ksel.arg_tensor(3)  # Shape b, l, s
        s_desc = ksel.arg_tensor(4)

        q_bs = q_desc.t.shape[:-2]
        k_bs = k_desc.t.shape[:-2]
        v_bs = v_desc.t.shape[:-2]

        bs = len(q_bs)

        print(q_bs)
        # torch._check(len(q_bs) == 2, lambda: f"TODO: batch dims {bs} not supported")

        q_l, q_d = q_desc.t.shape[-2:]
        k_s, k_d = k_desc.t.shape[-2:]
        v_s, v_e = v_desc.t.shape[-2:]

        torch._check(
            q_desc.t.dtype.is_floating_point
            and k_desc.t.dtype.is_floating_point
            and v_desc.t.dtype.is_floating_point
            and s_desc.t.dtype.is_floating_point,
            lambda: f"flash_attention: Expected floating point",
        )

        for q_b, k_b, v_b in zip(q_bs, k_bs, v_bs):
            torch._check(
                q_b == k_b and q_b == v_b,
                lambda: f"expected matching batch dims: {q_b}, {k_b}, {v_b}",
            )

        torch._check(q_d == k_d, lambda: f"expected matching qk features: {q_d}, {k_d}")

        torch._check(k_s == v_s, lambda: f"expected matching kv length: {q_d}, {k_d}")

        q_desc.specialize_dims(0, -1)
        k_desc.specialize_dims(0, -1)
        v_desc.specialize_dims(0, -1)
        a_desc.specialize_dims(0)
        print(q_desc)
        print(a_desc)

        # Result 0: Shape batch..., m, n
        ksel.return_new_tensor((*q_bs, q_l, v_e), dtype=torch.float16).specialize_dims(
            0, 2
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
        a_tensor_type = RankedTensorType(a.type)

        b, l, d = q_tensor_type.shape
        b, s, e = v_tensor_type.shape
        l = "?"
        s = "?"

        i_type_str = str(q_tensor_type.element_type)
        scale_type_str = str(scale_tensor_type.element_type)
        o_type_str = "f16"

        kwargs = {
            "b": b,
            "l": l,
            "d": d,
            "s": s,
            "e": e,
            "i_type": i_type_str,
            "scale_type": scale_type_str,
            "o_type": o_type_str,
        }
        template_file = "flash_attention_llm.mlir"
        target_function_name = f"sharktank_llm_flash_attention_{b}_{d}_{e}_{i_type_str}_{scale_type_str}_{o_type_str}"
        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(*call_function(target_function, q, k, v, scale, a))
        pass
