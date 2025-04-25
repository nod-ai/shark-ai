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
from sharktank.ops.signatures import scaled_dot_product_attention


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
        ops.attention_impls.register_attention_override_by_name(
            "masked_flash_attention"
        )
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
                return ops.scaled_dot_product_attention(
                    q, k, v, a=mask, is_causal=None, scale=scale
                )

        mod = MyModule()
        dtype = torch.dtype
        ep = torch.export.export(
            mod,
            args=(q, k, v, mask, scale),
            dynamic_shapes=dynamic_shapes,
        )
        output = aot.export(ep)
        output.verify()
        scaled_dot_product_attention.remove_override("masked_flash_attention")


class wave_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @parameterized.expand(
        [
            (1e-3, 1e-5),
            (1e-3, 1e-5),
            (1e-3, 1e-5),
        ]
    )
    def testWaveAttentionCausal(self, atol, rtol):
        dtype = torch.float16
        q = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        k = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        v = (torch.rand([4, 32, 128, 128]) * 64).to(dtype)
        result = kernels.wave_flash_attention(q, k, v)

        # Dequantize and test with normal matmul.
        # Tolerances are empirical and results are not expected to match exactly.
        ref = torch.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
