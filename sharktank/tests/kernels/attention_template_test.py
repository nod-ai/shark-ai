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
from sharktank import kernels
from sharktank.types import layout_utils
from sharktank.utils import debugging
from sharktank import ops


class custom_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(420)

    @parameterized.expand(
        [
            (torch.float32, 5e-3, 1e-3, True),
            (torch.float16, 5e-3, 1e-3, True),
            # Currently failing on unmasked error
            # (torch.float32, 5e-3, 1e-3, False),
            # (torch.float16, 5e-3, 1e-3, False),
        ]
    )
    def test_compare_torch_spda(self, dtype, atol, rtol, use_mask):
        H = 4  # Head dim
        N = 2  # Batch Size
        L = 256  # Target Seq Len
        S = 8  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask is same type as inputs, therefore its added to score
        mask = None
        scale = torch.tensor(1.0, dtype=dtype)
        res2 = ops.attention_impls.flash_attention(q, k, v, None, False, scale)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, mask, scale=scale
        )

        torch.testing.assert_close(res2.to(dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float32, True, False),
            # (torch.float32, False, True),
            # (torch.float16, True, True),
            # (torch.float8_e4m3fnuz, False, True),
        ]
    )
    def test_export_dynamic(self, dtype, static, use_mask):
        cast = False
        if dtype == torch.float8_e4m3fnuz:
            dtype = torch.float32
            cast = True
        H = 4  # Head dim
        N = 3  # Batch Size
        L = 19  # Target Seq Len
        S = 19  # Source Seq Len
        Eqk = Ev = 64  # embedding dimensions with subscript identifiers

        q = torch.rand([N, H, L, Eqk], dtype=dtype)
        k = torch.rand([N, H, S, Eqk], dtype=dtype)
        v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask is same type as inputs, therefore its added to score
        mask = torch.rand([L, S], dtype=dtype)
        if cast:
            q = q.to(torch.float8_e4m3fnuz)
            k = q.to(torch.float8_e4m3fnuz)
            v = v.to(torch.float8_e4m3fnuz)
            if use_mask:
                mask = mask.to(torch.float8_e4m3fnuz)
        scale = torch.tensor(1.0, dtype=dtype)
        dynamic_shapes = None
        if not static:
            L_dim = torch.export.Dim("L")
            S_dim = torch.export.Dim("S")
            dynamic_shapes = {
                "q": {2: L_dim},
                "k": {2: S_dim},
                "v": {2: S_dim},
                "mask": {},
                "scale": {},
            }
            if use_mask:
                dynamic_shapes["mask"] = {0: L_dim, 1: S_dim}

        class MyModule(torch.nn.Module):
            def forward(self, q, k, v, mask, scale):
                return ops.attention_impls.flash_attention(q, k, v, None, False, scale)

        mod = MyModule()
        dtype = torch.dtype
        if use_mask:
            ep = torch.export.export(
                mod,
                args=(q, k, v, mask, scale),
                dynamic_shapes=dynamic_shapes,
            )
        else:
            ep = torch.export.export(
                mod,
                args=(q, k, v, mask, scale),
                dynamic_shapes=dynamic_shapes,
            )
        output = aot.export(ep)
        output.verify()


if __name__ == "__main__":
    unittest.main()
