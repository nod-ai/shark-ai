# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Adapted from the Alibaba Wan team's original wan2.1 transformer implementation: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py

import math
from copy import copy
from collections import defaultdict, OrderedDict
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from sharktank.models.wan.attention import attention
from sharktank.models.wan.layers import *

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

from iree.turbine.ops.iree import trace_tensor
import os

__all__ = [
    "WanConfig",
    "WanModel",
]
################################################################################
# Models
################################################################################


@dataclass(kw_only=True)
class WanConfig(ModelConfig):
    # Wan Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
    wan_model_type: str = "t2v"
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
        super().__post_init__()

    @classmethod
    def _get_wan_config(cls: type["WanConfig"]) -> dict[str, Any]:
        return WanConfig.asdict()

    @classmethod
    def from_diffusers_config(cls: type["WanConfig"], config: dict) -> "WanConfig":
        params = {}
        default_dict = cls._get_wan_config()
        for param in default_dict.keys():
            if param in config.keys():
                params[param] = config[param]
            else:
                print(
                    f"Warning: Wan2.1 model config did not receive an entry for {param}. Using default {default_dict[param]}"
                )
                params[param] = default_dict[param]

        return WanConfig(**params)

    def to_hugging_face_properties(self) -> dict[str, Any]:
        hparams = {
            "dim": self.dim,
            "eps": self.eps,
            "ffn_dim": self.ffn_dim,
            "freq_dim": self.freq_dim,
            "in_dim": self.in_dim,
            "model_type": self.wan_model_type,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "out_dim": self.out_dim,
            "text_len": self.text_len,
        }
        return {"hparams": hparams}

    @classmethod
    def translate_hugging_face_config_dict_into_init_kwargs(
        cls, properties: dict[str, Any], /
    ) -> dict[str, Any]:
        if "hparams" in properties:
            properties = properties["hparams"]

        return {
            "dim": properties["dim"],
            "eps": properties["eps"],
            "ffn_dim": properties["ffn_dim"],
            "freq_dim": properties["freq_dim"],
            "in_dim": properties["in_dim"],
            "wan_model_type": properties["model_type"],
            "num_heads": properties["num_heads"],
            "num_layers": properties["num_layers"],
            "out_dim": properties["out_dim"],
            "text_len": properties["text_len"],
        }

    @classmethod
    def translate_hugging_face_config_into_init_kwargs(
        cls: type["WanConfig"],
        /,
        repo_id: str,
        revision: str | None = None,
        subfolder: str | None = None,
    ) -> dict[str, Any]:
        # There are 2 sets of parameters and the ones we use don't have a config.
        # We resort to using the config for the diffusers.WanTransformer3DModel.
        if subfolder is None:
            subfolder = "transformer"
        else:
            subfolder = f"{subfolder}/transformer"
        return super(cls, cls).translate_hugging_face_config_into_init_kwargs(
            repo_id, revision, subfolder
        )

    @classmethod
    def from_hugging_face_properties(
        cls: type["WanConfig"], properties: dict[str, Any]
    ) -> "WanConfig":
        return WanConfig(
            **cls.translate_hugging_face_config_dict_into_init_kwargs(properties)
        )

    @classmethod
    def get_wan_config(cls: type["WanConfig"]) -> "WanConfig":
        return WanConfig(**cls._get_wan_config())


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanModel(ThetaLayer):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    Adapted from c.ai implementation
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    def __init__(
        self, params: WanConfig, theta: Theta | None = None, dtype=torch.bfloat16
    ):
        r"""
        Initialize the diffusion model backbone.
        """

        # super().__init__()
        super().__init__(
            config=params,
            theta=theta,
        )

        self.wan_model_type = params.wan_model_type
        assert self.wan_model_type in ["t2v", "i2v"]
        self.params = copy(params)

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
        self.vae_z_dim = 16 if params.wan_model_type == "t2v" else 18
        self.sp_size = 1
        self.eps = params.eps

        self.patch_embedding = Conv3DLayer(
            self.theta("patch_embedding"),
            padding=(0, 0, 0),
            stride=tuple(params.patch_size),
        )
        self.text_embedding = TextTimeFFNEmbedder(self.theta("text_embedding"))
        self.time_embedding = TextTimeFFNEmbedder(self.theta("time_embedding"))
        self.time_projection = TimeGuidanceProjector(self.theta("time_projection"))

        # blocks
        cross_attn_type = f"{params.wan_model_type}_cross_attn"
        self.cross_attn_type = cross_attn_type
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    self.theta("blocks", i),
                    cross_attn_type,
                    params.dim,
                    params.ffn_dim,
                    params.num_heads,
                    params.window_size,
                    params.qk_norm,
                    params.cross_attn_norm,
                    params.eps,
                    self.dtype,
                )
                for i in range(params.num_layers)
            ]
        )

        # head
        self.head = Head(
            self.theta("head"),
            params.dim,
            params.out_dim,
            params.patch_size,
            params.eps,
            self.dtype,
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        if "i2v" in params.wan_model_type:
            self.img_emb = MLPProj(self.theta("img_emb"), 1280, params.dim)

    # Turbine's MLIR exporter requires all inputs to be tensors, not Python scalars, so init arg seperately
    def set_export_config(self, height, width, frame_num):
        target_shape = (
            self.vae_z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        self.seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

    @classmethod
    def from_config(cls, config: ModelConfig, /) -> "BaseLayer":
        """TODO: rename __init__'s arg params -> config and remove this method"""
        return cls(params=config)

    @classmethod
    def config_type(cls) -> type[WanConfig]:
        return WanConfig

    def sample_inputs(
        self,
        batch_size: int = 1,
        function: str = "forward",
        num_frames: int = 81,
        height: int = 512,
        width: int = 512,
    ) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:
        match self.wan_model_type:
            case "t2v":
                model_input = self._get_noise(
                    batch_size,
                    num_frames,
                    height,
                    width,
                    self.dtype,
                )
                context_shape = (batch_size, 28, 4096)
                args = tuple()
                kwargs = OrderedDict(
                    (
                        ("x", model_input.cpu()),
                        ("t", torch.tensor([999], dtype=self.dtype).cpu()),
                        ("context", torch.rand(context_shape, dtype=self.dtype).cpu()),
                    )
                )
                return args, kwargs
            case "i2v":
                model_input = self._get_noise(
                    batch_size,
                    num_frames,
                    height,
                    width,
                    self.dtype,
                )
                context_shape = (batch_size, 512, 4096)
                clip_fea_shape = (batch_size, 257, 1280)
                args = tuple()
                kwargs = OrderedDict(
                    (
                        ("x", model_input.cpu()),
                        ("t", torch.tensor([999], dtype=self.dtype).cpu()),
                        ("context", torch.rand(context_shape, dtype=self.dtype).cpu()),
                        (
                            "clip_fea",
                            torch.rand(clip_fea_shape, dtype=self.dtype).cpu(),
                        ),
                        ("y", model_input.clone().cpu()),
                    )
                )
                return args, kwargs

    def _get_noise(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float16,
    ):
        F = int(num_frames)
        noise = torch.randn(
            self.vae_z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            # allow for packing
            height // self.vae_stride[1],
            width // self.vae_stride[2],
            dtype=dtype,
        ).unsqueeze(0)
        if batch_size > 1:
            noise = noise.repeat(batch_size, 0, 0, 0, 0)
        return noise

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
            x (Tensor):
                Input video tensor batch, each with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (Tensor):
                Batch of text embeddings each with shape [B, L, C]
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (Tensor, *optional*):
                Conditional video input batch for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        seq_len = self.seq_len
        x = list(torch.unbind(x, dim=0))
        y = list(torch.unbind(y, dim=0))
        context = list(torch.unbind(context, dim=0))

        if "i2v" in self.wan_model_type:
            assert clip_fea is not None and y is not None

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  ## list of b c f h w

        grid_sizes = [list(u.shape[2:]) for u in x[:1]]
        x = [u.flatten(2).transpose(1, 2) for u in x]  ## 1 l c
        seq_lens = torch.tensor([u.size(1) for u in x[:1]], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        x = [
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ]

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type(self.dtype)
        ).type(self.dtype)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).type(self.dtype)
        assert e.dtype == self.dtype and e0.dtype == self.dtype

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.cat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )
        for idx, block in enumerate(self.blocks):
            x = block(x=x, **kwargs)

        # head
        x = [self.head(z, e.type(self.dtype)) for z in x]

        # unpatchify
        x = [self.unpatchify([z], grid_sizes)[0] for z in x]  ## list of b c f h w

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
        for u, v in zip(x, grid_sizes):
            # view [21, 32, 32, 1, 2, 2, 16]
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
