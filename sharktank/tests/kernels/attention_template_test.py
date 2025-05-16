# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

from safetensors import safe_open
import torch
from sharktank.utils.vmfb_runner import vmfbRunner

from iree.turbine import aot
from sharktank import kernels
from sharktank.types import layout_utils
from sharktank.utils import debugging
from sharktank import ops
from sharktank.ops.signatures import scaled_dot_product_attention
from safetensors import safe_open
from sharktank.utils.iree import (
    iree_to_torch,
    with_iree_device_context,
    torch_tensor_to_device_array,
)


class custom_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(420)

    #    @parameterized.expand(
    #        [
    #            (torch.float8_e4m3fnuz, 5e-3, 1e-3, False),
    #        ]
    #    )
    #    def test_compare_torch_spda(self, dtype, atol, rtol, use_mask):
    #        H = 4  # Head dim
    #        N = 3  # Batch Size
    #        L = 7  # Target Seq Len
    #        S = 6  # Source Seq Len
    #        Eqk = Ev = 64  # embedding dimensions with subscript identifiers
    #        cast = False
    #        if dtype == torch.float8_e4m3fnuz:
    #           cast = True
    #           dtype = torch.float32
    #        q = torch.rand([N, H, L, Eqk], dtype=dtype)
    #        k = torch.rand([N, H, S, Eqk], dtype=dtype)
    #        v = torch.rand([N, H, S, Ev], dtype=dtype)
    #        if cast:
    #            q = q.to(torch.float8_e4m3fnuz)
    #            k = k.to(torch.float8_e4m3fnuz)
    #            v = v.to(torch.float8_e4m3fnuz)
    #        # mask is same type as inputs, therefore its added to score
    #        mask = torch.zeros([L, S], dtype=dtype)
    #        scale = torch.tensor(1.0, dtype=dtype)
    #        if use_mask:
    #            mask = torch.rand([L, S], dtype=dtype)
    #
    #        res2 = kernels.masked_flash_attention(q, k, v, mask, scale=scale)
    #        # TODO: enable once unmasked kernel is fixed
    #        # res2 = kernels.flash_attention(q, k, v, scale)
    #        attn_weights = ops.matmul(
    #            q.to(torch.float32), k.transpose(2, 3).to(torch.float32)
    #        )
    #        attn_weights = attn_weights / math.sqrt(128)
    #
    #        # Flash attention.
    #        if softcap is not None:
    #            attn_weights = softcap * torch.tanh(attn_weights / softcap)
    #
    #        # Apply attention mask.
    #        if mask is None:
    #            mask = torch.full(
    #                (attn_weights.shape[2], attn_weights.shape[3]), float("-inf")
    #            )
    #            mask = torch.triu(mask, diagonal=1)[None, None, :, :]
    #            attn_weights = attn_weights + mask
    #        else:
    #            attn_weights = attn_weights + mask
    #
    #        attn_weights = ops.softmax(
    #            ops.to(attn_weights, dtype=torch.float32), dim=-1
    #        )
    #        attn_weights = (
    #            probs_quantizer.quantize(attn_weights).unpack().dequant()
    #        )
    #        attn_weights = ops.to(attn_weights, dtype=q.dtype)
    #        ref = ops.matmul(attn_weights, v)  # (bs, heads, slen, head_dim)
    #        ref = torch.nn.functional.scaled_dot_product_attention(
    #            q, k, v, mask, scale=scale
    #        )
    #
    #        torch.testing.assert_close(res2.to(dtype), ref, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            (torch.float8_e4m3fnuz, False, True, 19),
            # (torch.float8_e4m3fnuz, False, False, 2000),
            # (torch.float8_e4m3fnuz, True, True, 8),
            # (torch.float8_e4m3fnuz, True, False, 3),
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
        H = 32  # Head dim
        N = 16  # Batch Size
        L = SL  # Target Seq Len
        S = SL  # Source Seq Len
        Eqk = Ev = 128  # embedding dimensions with subscript identifiers

        tensors = {}
        with safe_open("inputs.st", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        with safe_open("outputs.st", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        q = tensors["q"]
        k = tensors["k"]
        v = tensors["v"]
        mask = tensors["mask"][0, 0, ...]
        # q = torch.rand([N, H, L, Eqk], dtype=dtype)
        # k = torch.rand([N, H, S, Eqk], dtype=dtype)
        # v = torch.rand([N, H, S, Ev], dtype=dtype)
        # mask = torch.zeros([L, S], dtype=dtype)
        # if use_mask:
        #    # mask is same type as inputs, therefore its added to score
        #    mask = torch.rand([L, S], dtype=dtype)
        # if cast:
        #    q = q.to(torch.float8_e4m3fnuz)
        #    k = k.to(torch.float8_e4m3fnuz)
        #    v = v.to(torch.float8_e4m3fnuz)
        #    if use_mask:
        #        mask = mask.to(torch.float8_e4m3fnuz)
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
                scale = torch.tensor(1.0, dtype=torch.float32)
                result = ops.attention_impls.masked_flash_attention(
                    q, k, v, mask, scale
                )
                return result  # /224.8484

        mod = MyModule()
        dtype = torch.dtype
        ep = torch.export.export(
            mod,
            args=(q, k, v, mask),
            dynamic_shapes=dynamic_shapes,
        )
        output = aot.export(ep)
        output.verify()
        output.save_mlir("testdata.mlir")
        scaled_dot_product_attention.remove_override("masked_flash_attention")
        output.session.set_flags("--iree-hip-target=gfx942")
        output.compile(save_to="testdata.vmfb", target_backends="rocm")
        vmfb_runner = vmfbRunner(
            device="hip",
            vmfb_path="testdata.vmfb",
        )

        device = vmfb_runner.config.device
        output = vmfb_runner.ctx.modules.module[f"main"](
            torch_tensor_to_device_array(q, device),
            torch_tensor_to_device_array(k, device),
            torch_tensor_to_device_array(v, device),
            torch_tensor_to_device_array(mask, device),
        )
        torch_ver = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=mask, is_causal=False, scale=None
        )
        torch.testing.assert_close(iree_to_torch(output)[0], torch_ver)
        print(torch.where(iree_to_torch(output)[0] > 0))
        print(torch.where(tensors["outputs"] > 0))
        old_values = {"old": iree_to_torch(output)[0]}
        # from safetensors.torch import save_file
        # save_file(old_values, "old.st")
        with safe_open("old.st", "pt") as f:
            old_value = f.get_tensor("old")

        # torch.testing.assert_close(
        #    old_value, iree_to_torch(output)[0]
        # )
        torch.testing.assert_close(iree_to_torch(output)[0], tensors["outputs"])


# --iree-opt-level=O3   --iree-hal-indirect-command-buffers=true   --iree-stream-resource-memory-model=discrete   --iree-hal-memoization=true


if __name__ == "__main__":
    unittest.main()
