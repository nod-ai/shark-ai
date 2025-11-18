# Copyright 2024 Black Forest Labs. Inc. and Flux Authors
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MMDIT Layers adapted from black-forest-labs' flux implementation
https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
"""
import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from amdsharktank import ops
from amdsharktank.types import unbox_tensor
from .base import ThetaLayer
from .linear import LinearLayer
from .modulation import ModulationLayer
from .norm import RMSNormLayer


def qk_norm(q, k, v, rms_q, rms_k):
    return rms_q(q).to(v), rms_k(k).to(v)


# TODO: Work on unifying with the current RoPE layer
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.to(torch.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(torch.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q, k, v, pe):
    q, k = apply_rope(unbox_tensor(q), unbox_tensor(k), unbox_tensor(pe))  # todo

    x = ops.scaled_dot_product_attention(q=q, k=k, v=v, a=None)
    x = ops.permute(x, (0, 2, 1, 3))
    x = x.reshape(x.shape[0], x.shape[1], -1)

    return x


class MMDITDoubleBlock(ThetaLayer):
    def __init__(self, theta, num_heads: int, hidden_size: int):
        super().__init__(theta)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_module("img_mod", ModulationLayer(theta("img_mod"), double=True))
        self.add_module("img_attn_qkv", LinearLayer(theta("img_attn.qkv")))
        self.add_module(
            "img_attn_norm_q",
            RMSNormLayer(
                theta("img_attn.norm.query_norm"), weight_name="scale", epsilon=1e-6
            ),
        )
        self.add_module(
            "img_attn_norm_k",
            RMSNormLayer(
                theta("img_attn.norm.key_norm"), weight_name="scale", epsilon=1e-6
            ),
        )
        self.add_module("img_attn_proj", LinearLayer(theta("img_attn.proj")))

        self.add_module("img_mlp1", LinearLayer(theta("img_mlp.0")))
        self.add_module("img_mlp2", LinearLayer(theta("img_mlp.2")))

        self.add_module("txt_mod", ModulationLayer(theta("txt_mod"), double=True))
        self.add_module("txt_attn_qkv", LinearLayer(theta("txt_attn.qkv")))
        self.add_module(
            "txt_attn_norm_q",
            RMSNormLayer(
                theta("txt_attn.norm.query_norm"), weight_name="scale", epsilon=1e-6
            ),
        )
        self.add_module(
            "txt_attn_norm_k",
            RMSNormLayer(
                theta("txt_attn.norm.key_norm"), weight_name="scale", epsilon=1e-6
            ),
        )
        self.add_module("txt_attn_proj", LinearLayer(theta("txt_attn.proj")))

        self.add_module("txt_mlp1", LinearLayer(theta("txt_mlp.0")))
        self.add_module("txt_mlp2", LinearLayer(theta("txt_mlp.2")))

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = ops.layer_norm(
            img, normalized_shape=(self.hidden_size,), weight=None, bias=None, eps=1e-6
        )
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn_qkv(img_modulated)
        img_qkv_2 = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
        )  #
        img_qkv_3 = ops.permute(img_qkv_2, (2, 0, 3, 1, 4))
        img_q, img_k, img_v = img_qkv_3
        img_q, img_k = qk_norm(
            img_q, img_k, img_v, self.img_attn_norm_q, self.img_attn_norm_k
        )

        # prepare text for attention
        txt_modulated = ops.layer_norm(
            txt, normalized_shape=(self.hidden_size,), weight=None, bias=None, eps=1e-6
        )
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_qkv_2 = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
        )  #
        txt_qkv_3 = ops.permute(txt_qkv_2, (2, 0, 3, 1, 4))
        txt_q, txt_k, txt_v = txt_qkv_3
        txt_q, txt_k = qk_norm(
            txt_q, txt_k, txt_v, self.txt_attn_norm_q, self.txt_attn_norm_k
        )

        # run actual attention
        q = ops.cat((txt_q, img_q), dim=2)
        k = ops.cat((txt_k, img_k), dim=2)
        v = ops.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the image blocks
        # TODO: Refactor this for code reuse with the txt blocks
        img = img + img_mod1.gate * self.img_attn_proj(img_attn)
        img_mlp_in = (1 + img_mod2.scale) * ops.layer_norm(
            img, normalized_shape=(self.hidden_size,), weight=None, bias=None, eps=1e-6
        ) + img_mod2.shift
        img_mlp_out1 = self.img_mlp1(img_mlp_in)
        img_mlp_out2 = ops.elementwise(
            functools.partial(F.gelu, approximate="tanh"), img_mlp_out1
        )
        img_mlp_out3 = self.img_mlp2(img_mlp_out2)
        img = img + img_mod2.gate * img_mlp_out3

        # calculate the text blocks
        txt = txt + txt_mod1.gate * self.txt_attn_proj(txt_attn)
        txt_mlp_in = (1 + txt_mod2.scale) * ops.layer_norm(
            txt, normalized_shape=(self.hidden_size,), weight=None, bias=None, eps=1e-6
        ) + txt_mod2.shift
        txt_mlp_out1 = self.txt_mlp1(txt_mlp_in)
        # TODO: Unify with modulation layer by taking act_fn as an arg
        txt_mlp_out2 = ops.elementwise(
            functools.partial(F.gelu, approximate="tanh"), txt_mlp_out1
        )
        txt_mlp_out3 = self.txt_mlp2(txt_mlp_out2)
        txt = txt + txt_mod2.gate * txt_mlp_out3

        return img, txt


class MMDITSingleBlock(ThetaLayer):
    def __init__(self, theta, num_heads: int, hidden_size: int, mlp_ratio: float):
        super().__init__(theta)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.add_module("mod", ModulationLayer(theta("modulation"), double=False))
        self.add_module(
            "attn_norm_q",
            RMSNormLayer(theta("norm.query_norm"), weight_name="scale", epsilon=1e-6),
        )
        self.add_module(
            "attn_norm_k",
            RMSNormLayer(theta("norm.key_norm"), weight_name="scale", epsilon=1e-6),
        )

        self.add_module("linear1", LinearLayer(theta("linear1")))
        self.add_module("linear2", LinearLayer(theta("linear2")))

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        mod, _ = self.mod(vec)
        x_norm = ops.layer_norm(
            x, normalized_shape=(self.hidden_size,), weight=None, bias=None, eps=1e-6
        )
        x_mod = (1 + mod.scale) * x_norm + mod.shift
        x_lin = self.linear1(x_mod)
        qkv, mlp = torch.split(
            x_lin, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        qkv_2 = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1)  #
        qkv_3 = ops.permute(qkv_2, (2, 0, 3, 1, 4))
        q, k, v = qkv_3
        q, k = qk_norm(q, k, v, self.attn_norm_q, self.attn_norm_k)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        gelu = ops.elementwise(functools.partial(F.gelu, approximate="tanh"), mlp)
        output = self.linear2(ops.cat((attn, gelu), 2))
        return x + mod.gate * output
