# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from sharktank.models.wan.attention import attention

from typing import Any, Optional, List
from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from sharktank.layers import *
from sharktank.layers.rotary_embedding_hf import select_concat
from sharktank.types import *
from sharktank.utils.create_cache import *
from sharktank import ops
from sharktank.ops.signatures import gelu_tanh_approximation

__all__ = [
    "TextTimeFFNEmbedder",
    "TimeGuidanceProjector",
    "WanAttentionBlock",
    "Head",
    "MLPProj",
]


def outer_hacked(a, b):
    return a.unsqueeze(1) * b.unsqueeze(0)


# This layer norm is used when non weight provided in irpa.
def layer_norm(inp: torch.Tensor, eps=1e-6):

    return ops.layer_norm(
        inp.float(), normalized_shape=(inp.shape[-1],), weight=None, bias=None, eps=eps
    ).type_as(inp)


class WanRotaryPositionalEmb(BaseLayer):
    def __init__(self, head_dim: int = 128):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = 10000
        self.interleaved = True

    def rope_precompute_sincos(self, x, grid_sizes, freqs, dtype):
        n, c = x.size(2), x.size(3) // 2

        # Split frequency components safely
        split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
        c1 = split_sizes[0]
        c2 = split_sizes[1]
        freq_f = freqs.narrow(1, 0, c1)
        freq_h = freqs.narrow(1, c1, c2)
        freq_w = freqs.narrow(1, c1 + c2, c2)

        output_cos = []
        output_sin = []
        # Loop over batch dimension
        for i in range(x.size(0)):
            # Access grid sizes as tensor elements and convert to Python ints
            f, h, w = grid_sizes[i]
            seq_len = f * h * w

            # Process complex numbers manually (avoid view_as_complex)
            x_slice = x[i, :seq_len].reshape(seq_len, n, -1, 2)
            real, imag = x_slice.unbind(-1)  # Split real/imaginary parts

            # Construct frequency tensors using Python ints for slicing
            f_freq = freq_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1)
            h_freq = freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1)
            w_freq = freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)

            # Combine frequencies and split into cos/sin components
            freqs_combined = torch.cat([f_freq, h_freq, w_freq], dim=-1)
            cos_theta = freqs_combined[..., :c].reshape(seq_len, 1, -1)
            sin_theta = freqs_combined[..., c:].reshape(seq_len, 1, -1)
            output_cos.append(cos_theta.to(dtype=dtype))
            output_sin.append(sin_theta.to(dtype=dtype))
        return *output_cos, *output_sin

    def apply(self, q, k, sincos_cache):
        def rope_apply(x):
            output = []
            cos_theta, sin_theta = sincos_cache
            # Loop over batch dimension

            for i in range(x.size(0)):

                # # Process complex numbers manually (avoid view_as_complex)
                real = x[..., 0 : self.head_dim : 2]
                imag = x[..., 1 : self.head_dim : 2]

                # Manual rotation (real and imaginary parts)
                new_real = real * cos_theta - imag * sin_theta
                new_imag = real * sin_theta + imag * cos_theta

                # Recombine and concatenate with remaining elements
                cated = select_concat(new_real, new_imag)
                # Collapse the last two dimensions.
                cated = cated.flatten(start_dim=-2, end_dim=-1)
                output.append(cated.squeeze(0))
            return torch.stack(output)

        return rope_apply(q), rope_apply(k)


class TextTimeFFNEmbedder(ThetaLayer):
    def __init__(self, theta: Theta):
        super().__init__(theta)
        self.in_layer = LinearLayer(theta("0"))
        self.out_layer = LinearLayer(theta("2"))

    def forward(self, x: AnyTensor) -> AnyTensor:
        x = self.in_layer(x)
        x = gelu_tanh_approximation(x)
        return self.out_layer(x)


class TimeGuidanceProjector(ThetaLayer):
    def __init__(self, theta: Theta):
        super().__init__(theta)
        self.out_layer = LinearLayer(theta("1"))

    def forward(self, x: AnyTensor) -> AnyTensor:
        x = ops.elementwise(torch.nn.functional.silu, x)
        return self.out_layer(x)


class GuidanceEmbedding(ThetaLayer):
    def __init__(self, theta: Theta):
        super().__init__(theta)
        self.in_layer = LinearLayer(theta("0"))
        self.out_layer = LinearLayer(theta("2"))

    def forward(self, x: AnyTensor) -> AnyTensor:
        x = self.in_layer(x)
        x = ops.elementwise(torch.nn.functional.silu, x)
        return self.out_layer(x)


# TODO: Rework on top of shared RMSnorm layer.
class WanRMSNorm(ThetaLayer):
    def __init__(
        self,
        theta,
        weight_name: str = "weight",
        *,
        dim,
        eps=1e-5,
        dtype=torch.bfloat16,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        # self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(LayerNorm):
    def __init__(self, theta, dim, eps=1e-6, dtype=torch.bfloat16):
        # elementwise_affine is flag for training, so delete it.
        super().__init__(theta, eps=eps, normalized_shape=(dim,))
        self.dtype = dtype

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.type(self.dtype)).type_as(x)


class WanSelfAttention(ThetaLayer):
    def __init__(
        self,
        theta,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        dtype=torch.bfloat16,
    ):
        assert dim % num_heads == 0
        super().__init__(theta)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.dtype = dtype

        # layers
        self.q = LinearLayer(theta("q"))
        self.k = LinearLayer(theta("k"))
        self.v = LinearLayer(theta("v"))
        self.o = LinearLayer(theta("o"))
        self.norm_q = (
            WanRMSNorm(theta("norm_q"), dim=dim, eps=eps, dtype=self.dtype)
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            WanRMSNorm(theta("norm_q"), dim=dim, eps=eps, dtype=self.dtype)
            if qk_norm
            else nn.Identity()
        )
        self.rope = WanRotaryPositionalEmb(head_dim=self.head_dim)

    def forward(self, x, seq_lens, grid_sizes, freqs, x_kv=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x, x_kv=None):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            if x_kv is not None:
                k = self.norm_k(self.k(x_kv)).view(b, s, n, d)
                v = self.v(x_kv).view(b, s, n, d)
            else:
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x, x_kv=x_kv)

        sincos_cache = self.rope.rope_precompute_sincos(
            q, grid_sizes, freqs, dtype=self.dtype
        )
        q, k = self.rope.apply(q, k, sincos_cache)
        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            dtype=self.dtype,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v, dtype=self.dtype, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(
        self,
        theta,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        dtype=torch.bfloat16,
    ):
        super().__init__(theta, dim, num_heads, window_size, qk_norm, eps, dtype)

        self.k_img = LinearLayer(self.theta("k_img"))
        self.v_img = LinearLayer(self.theta("v_img"))
        self.norm_k_img = (
            WanRMSNorm(self.theta("norm_k_img"), dim=dim, eps=eps)
            if qk_norm
            else nn.Identity()
        )

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = attention(q, k_img, v_img, dtype=self.dtype, k_lens=None)
        x = attention(q, k, v, dtype=self.dtype, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(ThetaLayer):
    def __init__(
        self,
        theta,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        dtype=torch.bfloat16,
    ):
        super().__init__(theta)
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.dtype = dtype

        # layers
        self.self_attn = WanSelfAttention(
            self.theta("self_attn"), dim, num_heads, window_size, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(self.theta("norm3"), dim, eps)
            if cross_attn_norm
            else nn.Identity()
        )

        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            self.theta("cross_attn"), dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.ffn = TextTimeFFNEmbedder(self.theta("ffn"))

        # modulation
        self.modulation = self.theta_tensor("modulation")

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        assert e.dtype == self.dtype
        mod_e = (self.modulation + e).type(self.dtype)
        chunk_size = mod_e.size(1) // 6
        # workaround chunk lowering to as_strided
        e = [
            mod_e.narrow(1, i * chunk_size, chunk_size).type(self.dtype)
            for i in range(6)
        ]
        # self-attention
        x = ops.layer_norm(
            x[0], normalized_shape=(self.dim,), weight=None, bias=None, eps=1e-6
        )
        e_rhs = (1 + e[1]) + e[0]
        y = self.self_attn(x * e_rhs, seq_lens, grid_sizes, freqs)
        x = (x + y * e[2]).type(self.dtype)
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(layer_norm(x).type(self.dtype) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(ThetaLayer):
    def __init__(self, theta, dim, out_dim, patch_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__(theta)
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.dtype = dtype

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.head = LinearLayer(theta("head"))

        # modulation
        self.modulation = self.theta_tensor("modulation")

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == self.dtype
        """
        workaround chunk lowering to as_strided
        """
        mod_e = self.modulation + e.unsqueeze(1)
        chunk_size = mod_e.size(1) // 2
        e = [mod_e.narrow(1, i * chunk_size, chunk_size) for i in range(2)]
        x = (self.head(layer_norm(x) * (1 + e[1]) + e[0])).type(self.dtype)
        return x


class MLPProj(ThetaLayer):
    def __init__(self, theta, in_dim, out_dim):
        super().__init__(theta)
        self.add_module("proj_0", LayerNorm(theta("proj.0")))
        self.add_module("proj_1", LinearLayer(theta("proj.1")))
        self.add_module("proj_3", LinearLayer(theta("proj.3")))
        self.add_module("proj_4", LayerNorm(theta("proj.4")))

    def forward(self, image_embeds):
        image_embeds = self.proj_0(image_embeds)
        image_embeds = self.proj_1(image_embeds)
        image_embeds = ops.elementwise(torch.nn.functional.gelu, image_embeds)
        image_embeds = self.proj_3(image_embeds)
        clip_extra_context_tokens = self.proj_4(image_embeds)
        return clip_extra_context_tokens
