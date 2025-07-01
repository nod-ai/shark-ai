# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from sharktank.models.wan import attention

from typing import Any, Optional, List
from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from sharktank.layers import *
from sharktank.layers.rotary_embedding_v2 import select_concat
from sharktank.types import *
from sharktank.utils.create_cache import *
from sharktank import ops
from sharktank.ops.signatures import gelu_tanh_approximation

from iree.turbine.ops.iree import trace_tensor
import os

__all__ = [
    "WanParams",
    "WanModel",
]
################################################################################
# Models
################################################################################

DTYPE = torch.bfloat16

@dataclass(kw_only=True)
class WanParams(ModelConfig):
    # Wan Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
    wan_model_type: str = 't2v' 
    # 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # Fixed length for text embeddings
    text_len: int = 512
    # Input video channels (C_in)
    in_dim: int = 36
    # Hidden dimension of the transformer
    dim: int = 5120
    # Intermediate dimension in feed-forward network
    ffn_dim: int = 13824
    # Dimension for sinusoidal time embeddings
    freq_dim: int = 256
    # Input dimension for text embeddings
    text_dim: int = 4096
    # Output video channels (C_out)
    out_dim: int = 16
    # Number of attention heads
    num_heads: int = 40
    # Number of transformer blocks
    num_layers: int = 40
    # Window size for local attention (-1 indicates global attention)
    window_size: tuple[int, int] = (-1, -1)
    # Enable query/key normalization
    qk_norm: bool = True
    # Enable cross-attention normalization
    cross_attn_norm: bool = True
    # Epsilon value for normalization layers
    eps: float = 1e-6

    def __post_init__(self):
        self.model_type = WanModel
        self.wan_model_type = 't2v'
        super().__post_init__()
    
    @classmethod
    def _get_wan_params(cls: type["WanParams"]) -> dict[str, Any]:
        return {
                "patch_size": [1, 2, 2],
                "wan_model_type": "t2v",
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "num_heads": 40,
                "num_layers": 40,
                "in_dim": 36, 
                "window_size": [-1, -1],
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
        }
    
    @classmethod
    def from_diffusers_config(cls: type["WanParams"], config: dict) -> "WanParams":
        params = {}
        default_dict = cls._get_wan_params()
        for param in default_dict.keys():
            if param in config.keys():
                params[param] = config[param]
            else:
                print(f"Warning: Wan2.1 model config did not receive an entry for {param}. Using default {default_dict[param]}")
                params[param] = default_dict[param]

        return WanParams(**params)

    @classmethod
    def get_wan_params(cls: type["WanParams"]) -> "WanParams":
        return WanParams(
            **cls._get_wan_params()
        )

def outer_hacked(a, b):
    return a.unsqueeze(1) * b.unsqueeze(0)

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = outer_hacked(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs)

class WanRotaryPositionalEmb(BaseLayer):
    def __init__(
        self, head_dim: int = 128 
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = 10000
        self.interleaved = True


    def rope_precompute_sincos(self, x, grid_sizes, freqs, dtype):
        n, c = x.size(2), x.size(3) // 2

        # Split frequency components safely
        split_sizes = [c - 2*(c//3), c//3, c//3]
        c1 = split_sizes[0]
        c2 = split_sizes[1]
        freq_f= freqs.narrow(1, 0, c1)
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
                # x_slice = x[i, :seq_len].reshape(seq_len, n, 2, -1)
                # real, imag = x_slice.unbind(-2)  # Split real/imaginary parts
                real = x[..., 0 : self.head_dim : 2]
                imag = x[..., 1 : self.head_dim : 2]

                # Manual rotation (real and imaginary parts)
                new_real = real * cos_theta - imag * sin_theta
                new_imag = real * sin_theta + imag * cos_theta

                # rotated = torch.stack([new_real, new_imag], dim=-1).flatten(2)
                # rotated = torch.cat([rotated, x[i, seq_len:]], dim=0)
                # output.append(rotated)
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
        # RuntimeError: Unhandled FakeTensor Device Propagation for aten.mm.default, found two different devices cuda:0, cpu
        # device = next(self.parameters()).device
        # x = x.to(device)
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


'''
class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.bfloat16()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
'''
class WanRMSNorm(ThetaLayer):

    def __init__(
            self, 
            theta, 
            weight_name: str = "weight",
            *,
            dim, 
            eps=1e-5,
            dtype=DTYPE,
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
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

'''
class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.bfloat16()).type_as(x)
'''
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


# class WanSelfAttention(nn.Module):
class WanSelfAttention(ThetaLayer):

    def __init__(self,
                 theta,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 dtype=DTYPE):
        assert dim % num_heads == 0
        super().__init__(theta)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.dtype=dtype

        # layers
        '''
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        '''
        self.q = LinearLayer(theta("q"))
        self.k = LinearLayer(theta("k"))
        self.v = LinearLayer(theta("v"))
        self.o = LinearLayer(theta("o"))
        self.norm_q = WanRMSNorm(theta("norm_q"), dim=dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(theta("norm_q"), dim=dim, eps=eps) if qk_norm else nn.Identity()
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


        '''
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        '''
        # print("DEBUG grid_sizes: ", grid_sizes) # grid_sizes:  FakeTensor(..., size=(1, 3), dtype=torch.int64)
        # grid_sizes = [[f, h, w]] = [[21,32,32]] math.prod(grid_sizes[0])=21504
        # print("DEBUG freqs: ", freqs) # freqs:  tensor([[ 1.0000+0.0000e+00j,  1.0000+0.0000e+00j,  1.0000+0.0000e+00j, ... dtype=torch.complex128)
        # print("DEBUG freqs.shape: ", freqs.shape) # torch.Size([1024, 64])
        # print("DEBUG q: ", q) # q:  FakeTensor(..., size=(1, 21504, 40, 128), dtype=torch.bfloat16)
        sincos_cache = self.rope.rope_precompute_sincos(q, grid_sizes, freqs, dtype=self.dtype)
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
        '''
        x = flash_attention(q, k, v, k_lens=context_lens)
        '''
        x = attention(q, k, v, dtype=self.dtype, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    
class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 theta,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 dtype=DTYPE):
        super().__init__(theta, dim, num_heads, window_size, qk_norm, eps, dtype)

        '''
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        '''
        self.k_img = LinearLayer(self.theta("k_img"))
        self.v_img = LinearLayer(self.theta("v_img"))
        self.norm_k_img = WanRMSNorm(self.theta("norm_k_img"), dim=dim, eps=eps) if qk_norm else nn.Identity()

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

        '''
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        '''
        img_x = attention(q, k_img, v_img, dtype=self.dtype, k_lens=None)
        x = attention(q, k, v, dtype=self.dtype, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x



WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

# This layer norm is used when non weight provided in irpa.
def layer_norm(inp: torch.Tensor, eps=1e-6):
    return ops.layer_norm(
        inp.float(), normalized_shape=(inp.shape[-1],), weight=None, bias=None, eps=eps
    ).type_as(inp)

    
# class WanAttentionBlock(nn.Module):
class WanAttentionBlock(ThetaLayer):

    def __init__(self,theta,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 dtype=DTYPE,
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
        '''
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        '''
        self.self_attn = WanSelfAttention(self.theta("self_attn"), dim, num_heads, window_size, qk_norm,
                                          eps)
        '''
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # elementwise_affine is used for training
        '''
        self.norm3 = WanLayerNorm(self.theta("norm3"),
            dim, eps) if cross_attn_norm else nn.Identity()
       
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](self.theta("cross_attn"), 
                                                                    dim,
                                                                    num_heads,
                                                                    (-1, -1),
                                                                    qk_norm,
                                                                    eps)
        '''
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        '''
        self.ffn = TextTimeFFNEmbedder(self.theta("ffn"))

        # modulation
        '''
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        '''
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

        """
        workaround chunk lowering to as_strided
        """
        assert e.dtype == self.dtype
        mod_e = (self.modulation + e).type(self.dtype)
        chunk_size = mod_e.size(1) // 6
        e = [mod_e.narrow(1, i * chunk_size, chunk_size).type(self.dtype) for i in range(6)]
        # self-attention
        x = ops.layer_norm(x[0], normalized_shape=(self.dim,), weight=None, bias=None, eps=1e-6)
        e_rhs = (1 + e[1]) + e[0]
        y = self.self_attn(
            x * e_rhs, seq_lens, grid_sizes, freqs)
        x = (x + y * e[2]).type(self.dtype)
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(layer_norm(x).type(self.dtype) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

'''
class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.bfloat16
        with amp.autocast(dtype=torch.bfloat16):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x
'''
class Head(ThetaLayer):

    def __init__(self, theta, dim, out_dim, patch_size, eps=1e-6, dtype=DTYPE):
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
        '''
        workaround chunk lowering to as_strided
        '''
        mod_e = self.modulation + e.unsqueeze(1)
        chunk_size = mod_e.size(1) // 2
        e = [mod_e.narrow(1, i * chunk_size, chunk_size) for i in range(2)]
        x = (self.head(layer_norm(x) * (1 + e[1]) + e[0])).type(self.dtype)
        return x

'''

class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
'''
class MLPProj(ThetaLayer):

    def __init__(self, theta, in_dim, out_dim):
        super().__init__(theta)
        self.add_module(
            "proj_0", LayerNorm(theta("proj.0"))
        )
        self.add_module("proj_1", LinearLayer(theta("proj.1")))
        self.add_module("proj_3", LinearLayer(theta("proj.3")))
        self.add_module(
            "proj_4", LayerNorm(theta("proj.4"))
        )
        

    def forward(self, image_embeds):
        image_embeds = self.proj_0(image_embeds)
        image_embeds = self.proj_1(image_embeds)
        image_embeds = ops.elementwise(torch.nn.functional.gelu, image_embeds)
        image_embeds = self.proj_3(image_embeds)
        clip_extra_context_tokens = self.proj_4(image_embeds)
        return clip_extra_context_tokens

    
class WanModel(ThetaLayer):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    Adapted from c.ai implementation
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    def __init__(self, params: WanParams, theta: Theta | None = None, dtype = torch.bfloat16):
        r"""
        Initialize the diffusion model backbone.
        """

        # super().__init__()
        super().__init__(
            config=params,
            theta=theta,
        )

        self.wan_model_type = params.wan_model_type
        assert self.wan_model_type in ['t2v', 'i2v']

        self.patch_size = params.patch_size
        self.text_len = params.text_len
        self.in_dim = params.in_dim
        self.dim = params.dim
        self.ffn_dim = params.ffn_dim
        self.freq_dim = params.freq_dim
        self.text_dim = params.text_dim
        self.out_dim = params.out_dim
        self.num_heads = params.num_heads
        self.num_layers = params.num_layers
        self.window_size = params.window_size
        self.qk_norm = params.qk_norm
        self.cross_attn_norm = params.cross_attn_norm
        self.dtype = dtype
        self.vae_stride = (4, 8, 8)
        self.vae_z_dim = 16
        self.sp_size = 1
        self.eps = params.eps

        '''
        patch_embedding.weight, patch_embedding.bias
        self.patch_embedding = nn.Conv3d(
            params.in_dim, params.dim, kernel_size=params.patch_size, stride=params.patch_size)
        '''
        self.patch_embedding = Conv3DLayer(self.theta("patch_embedding"), padding=(0, 0, 0), stride=tuple(params.patch_size))
        '''
        self.text_embedding = nn.Sequential(
            LinearLayer(self.theta("text_embedding.0")), nn.GELU(approximate='tanh'),
            LinearLayer(self.theta("text_embedding.2")))
        '''
        self.text_embedding = TextTimeFFNEmbedder(self.theta("text_embedding"))
        
        '''
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        '''
        self.time_embedding = TextTimeFFNEmbedder(self.theta("time_embedding"))
        self.time_projection = TimeGuidanceProjector(self.theta("time_projection"))

        # blocks
        '''
        cross_attn_type = f'{model_type}_cross_attn'
        self.cross_attn_type = cross_attn_type
        '''
        cross_attn_type = f'{params.wan_model_type}_cross_attn'
        self.cross_attn_type = cross_attn_type
        self.blocks = nn.ModuleList([
            WanAttentionBlock(self.theta("blocks", i), cross_attn_type, params.dim, params.ffn_dim, params.num_heads,
                              params.window_size, params.qk_norm, params.cross_attn_norm, params.eps, self.dtype)
            for i in range(params.num_layers)
        ])


        # head
        self.head = Head(self.theta("head"), params.dim, params.out_dim, params.patch_size, params.eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        '''
        # if model_type == 'i2v':
        if 'i2v' in model_type:
            self.img_emb = MLPProj(1280, dim)
        '''
        if 'i2v' in params.wan_model_type:
            self.img_emb = MLPProj(self.theta("img_emb"), 1280, params.dim)

    # Turbine's MLIR exporter requires all inputs to be tensors, not Python scalars, so init arg seperately
    def set_export_config(self, height, width, frame_num):
        target_shape = (self.vae_z_dim, 
            (frame_num - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2]
        )
        self.seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

    @classmethod
    def from_config(cls, config: ModelConfig, /) -> "BaseLayer":
        """TODO: rename __init__'s arg params -> config and remove this method"""
        return cls(params=config)

    @classmethod
    def config_type(cls) -> type[WanParams]:
        return WanParams
        
    def _get_noise(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ):
        F = num_frames
        return [torch.randn(
            self.vae_z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            # allow for packing
            height // self.vae_stride[1],
            width // self.vae_stride[2],
            dtype=dtype,
        )] * batch_size
    
    def forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        seq_len = self.seq_len
        if "i2v" in self.wan_model_type:
            assert clip_fea is not None and y is not None

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] ## list of b c f h w

        ## hacked to only use the first one
        '''
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x[:1]])
        '''
        grid_sizes = [list(u.shape[2:]) for u in x[:1]]
        x = [u.flatten(2).transpose(1, 2) for u in x] ## 1 l c
        seq_lens = torch.tensor([u.size(1) for u in x[:1]], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        ## here i will loop the x instead to maintain as a list form
        x = [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type(self.dtype)).type(self.dtype)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).type(self.dtype)
        assert e.dtype == self.dtype and e0.dtype == self.dtype

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.cat([context_clip, context], dim=1)
        """
        only things to change is: 
        x is a list of tensors
        """
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        for idx, block in enumerate(self.blocks):
            x = block(x=x, **kwargs)

        # head
        x = [self.head(z, e.type(self.dtype)) for z in x]

        # unpatchify
        x = [self.unpatchify([z], grid_sizes)[0] for z in x] ## list of b c f h w

        return [u.squeeze(0).type(self.dtype) for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        '''
        for u, v in zip(x, grid_sizes.tolist()):
        '''
        for u, v in zip(x, grid_sizes):
            # view [21, 32, 32, 1, 2, 2, 16]
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
    
    def sample_inputs(
        self, batch_size: int = 1, function: Optional[str] = "forward"
    ) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:
        if not (function is None or function == "forward"):
            raise ValueError(f'Only function "forward" is supported. Got "{function}"')
        
        # Prepare inputs
        # input config
      
        # Get wan model input
        model_input = self.model._get_noise(
            batch_size,
            self.num_frames,
            self.height,
            self.width,
        )
        if function == "forward":
            context_shape = (28, 4096)
            args = tuple()
            kwargs = OrderedDict(
                (
                    ("x", model_input[0]),
                    ("t", torch.tensor([999], dtype=torch.float16)),
                    ("context", torch.rand(context_shape, dtype=torch.float16)),
                )
            )
        else:
            raise ValueError(f"Received invalid specifier for `function` to export: {function}")
        return args, kwargs