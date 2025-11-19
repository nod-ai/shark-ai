# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from iree.turbine import aot
from amdsharktank import kernels
from amdsharktank import ops
from amdsharktank.types import PlanarQuantizedTensor, TensorScaledLayout, unbox_tensor


def make_quantized(t: torch.Tensor, scale: torch.Tensor):
    return PlanarQuantizedTensor(
        shape=t.shape, layout=TensorScaledLayout(shape=t.shape, d=scale, qs=t)
    )


class custom_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(420)

    @parameterized.expand(
        [
            (torch.float32, 5e-3, 1e-3, True),
            (torch.float16, 5e-3, 1e-3, True),
            (torch.float32, 5e-3, 1e-3, False),
            (torch.float16, 5e-3, 1e-3, False),
        ]
    )
    def test_compare_torch_spda(self, dtype, atol, rtol, use_mask):
        H = 4  # Head dim
        N = 3  # Batch Size
        L = 7  # Target Seq Len
        S = 6  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask is same type as inputs, therefore its added to score
        mask = torch.zeros([L, S], dtype=dtype)
        scale = torch.tensor(1.0, dtype=dtype)
        if use_mask:
            mask = torch.rand([L, S], dtype=dtype)

        res2 = kernels.masked_flash_attention(q, k, v, mask, scale=scale)
        # TODO: enable once unmasked kernel is fixed
        # res2 = kernels.flash_attention(q, k, v, scale)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, mask, scale=scale
        )

        torch.testing.assert_close(res2.to(dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float8_e4m3fnuz, False, True, 19),
            (torch.float8_e4m3fnuz, False, False, 2000),
            (torch.float8_e4m3fnuz, True, True, 8),
            (torch.float8_e4m3fnuz, True, False, 3),
        ]
    )
    def test_export_custom_sdpa(self, dtype, static, use_mask, SL):
        cast = False
        # Get rid of this once output type is supported in sdpa op
        if dtype == torch.float8_e4m3fnuz:
            dtype = torch.float32
            cast = True
        H = 4  # Head dim
        N = 3  # Batch Size
        L = SL  # Target Seq Len
        S = SL  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        mask = torch.zeros([L, S], dtype=dtype)
        if use_mask:
            # mask is same type as inputs, therefore its added to score
            mask = torch.rand([L, S], dtype=dtype)
        if cast:
            q = q.to(torch.float8_e4m3fnuz)
            k = q.to(torch.float8_e4m3fnuz)
            v = v.to(torch.float8_e4m3fnuz)
        dynamic_shapes = None
        if not static:
            L_dim = torch.export.Dim("L")
            S_dim = torch.export.Dim("S")
            dynamic_shapes = {
                "q": {2: L_dim},
                "k": {2: S_dim},
                "v": {2: S_dim},
                "mask": {},
            }
            if use_mask:
                dynamic_shapes["mask"] = {0: L_dim, 1: S_dim}

        class MyModule(torch.nn.Module):
            def forward(self, q, k, v, mask):
                if cast:
                    q = make_quantized(q, scale=torch.tensor(1.0))
                    k = make_quantized(k, scale=torch.tensor(1.0))
                    v = make_quantized(v, scale=torch.tensor(1.0))

                scale = 1.0
                return unbox_tensor(
                    ops.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        a=mask,
                        is_causal=False,
                        scale=scale,
                        impl="amdsharktank",
                    )
                )

        mod = MyModule()
        dtype = torch.dtype
        ep = torch.export.export(
            mod,
            args=(q, k, v, mask),
            dynamic_shapes=dynamic_shapes,
        )
        output = aot.export(ep)
        mlir_src = str(output.mlir_module)
        assert "iree_linalg_ext.attention" in mlir_src
        output.verify()


if __name__ == "__main__":
    unittest.main()
