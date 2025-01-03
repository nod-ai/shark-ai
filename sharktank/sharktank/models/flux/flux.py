# Copyright 2024 Advanced Micro Devices, Inc.
# Copyright 2024 Black Forest Labs. Inc. and Flux Authors
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Model adapted from black-forest-labs' flux implementation
https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
"""

from typing import Any, Optional
from collections import OrderedDict
from copy import copy
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers import *
from ...types import *
from ...utils.create_cache import *
from ...utils.testing import make_rand_torch
from ... import ops

__all__ = [
    "FluxModelV1",
]

################################################################################
# Models
################################################################################


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

    @staticmethod
    def from_hugging_face_properties(properties: dict[str, Any]) -> "FluxParams":
        p = properties["hparams"]

        in_channels = p["in_channels"]
        out_channels = p["in_channels"]
        vec_in_dim = p["pooled_projection_dim"]
        context_in_dim = p["joint_attention_dim"]
        mlp_ratio = 4.0
        hidden_size = vec_in_dim * int(mlp_ratio)
        num_heads = p["num_attention_heads"]
        depth = p["num_layers"]
        depth_single_blocks = p["num_single_layers"]

        # TODO: figure out relation between hidden_size, num_heads and
        # attention_head_dim.
        # diffusers.FluxTransformer2DModel also hardcodes this.
        axes_dim = [16, 56, 56]
        assert sum(axes_dim) == p["attention_head_dim"]

        theta = 10_000
        qkv_bias = True
        guidance_embed = p["guidance_embeds"]

        return FluxParams(
            in_channels=in_channels,
            out_channels=out_channels,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            mlp_ratio=mlp_ratio,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            axes_dim=axes_dim,
            theta=theta,
            qkv_bias=qkv_bias,
            guidance_embed=guidance_embed,
        )


class FluxModelV1(ThetaLayer):
    """FluxModel adapted from Black Forest Lab's implementation."""

    def __init__(self, theta: Theta, params: FluxParams):
        super().__init__(
            theta,
        )

        self.params = copy(params)
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.add_module("img_in", LinearLayer(theta("img_in")))
        self.add_module("time_in", MLPEmbedder(theta("time_in")))
        self.add_module("vector_in", MLPEmbedder(theta("vector_in")))
        self.guidance = False
        if params.guidance_embed:
            self.guidance = True
            self.add_module("guidance_in", MLPEmbedder(theta("guidance_in")))
        self.add_module("txt_in", LinearLayer(theta("txt_in")))

        self.double_blocks = nn.ModuleList(
            [
                MMDITDoubleBlock(
                    theta("double_blocks", i),
                    self.num_heads,
                )
                for i in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMDITSingleBlock(
                    theta("single_blocks", i),
                    self.num_heads,
                )
                for i in range(params.depth_single_blocks)
            ]
        )

        self.add_module(
            "final_layer",
            LastLayer(theta("final_layer")),
        )

        self.dtype = self._deduce_dtype()

    def forward(
        self,
        img: AnyTensor,
        img_ids: AnyTensor,
        txt: AnyTensor,
        txt_ids: AnyTensor,
        timesteps: AnyTensor,
        y: AnyTensor,
        guidance: AnyTensor | None = None,
    ) -> AnyTensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def sample_inputs(
        self, batch_size: int = 1, function: Optional[str] = None
    ) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:
        if not (function is None or function == "forward"):
            raise ValueError(f'Only function "forward" is supported. Got "{function}"')

        # The allowed range of these values is dependent on the model size.
        # They will not work for all variants, specifically toy-sized models.
        output_img_height = 1024
        output_img_width = 1024
        output_img_channels = 3

        img = self._get_noise(
            batch_size, output_img_height, output_img_width, self.dtype
        )

        _, c, h, w = img.shape
        img = img.reshape(batch_size, h * w // 4, c * 4)

        img_ids = torch.zeros(h // 2, w // 2, output_img_channels)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = img_ids.reshape(1, h * w // 4, output_img_channels)
        img_ids = img_ids.repeat(batch_size, 1, 1)

        # T5 encoder output
        txt_context_length = 512
        txt_dims_per_token = 4096
        txt = torch.rand([1, txt_context_length, txt_dims_per_token], dtype=self.dtype)
        txt = txt.repeat(batch_size, 1, 1)
        txt_ids = torch.zeros(batch_size, txt.shape[1], output_img_channels)

        timesteps = torch.rand([batch_size], dtype=self.dtype)

        # CLIP text model output
        y = make_rand_torch([1, 768], dtype=self.dtype)
        y = y.repeat(batch_size, 1)

        args = tuple()
        kwargs = OrderedDict(
            (
                ("img", img),
                ("img_ids", img_ids),
                ("txt", txt),
                ("txt_ids", txt_ids),
                ("timesteps", timesteps),
                ("y", y),
            )
        )

        if self.guidance:
            kwargs["guidance"] = torch.full([batch_size], 3.5, dtype=self.dtype)

        return args, kwargs

    def _get_noise(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ):
        return torch.randn(
            batch_size,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            dtype=dtype,
        )

    def _deduce_dtype(self) -> torch.dtype:
        dtype = self.theta("img_in.weight").dtype
        assert (
            dtype == self.theta("time_in.in_layer.weight").dtype
        ), "Inconsistent dtype"
        return dtype


################################################################################
# Layers
################################################################################


# TODO: Refactor these functions to other files. Rope can probably be merged with
# our rotary embedding layer, some of these functions are shared with layers/mmdit.py
def timestep_embedding(
    t: AnyTensor, dim, max_period=10000, time_factor: float = 1000.0
):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def layer_norm(inp):
    weight = torch.ones(inp.shape)
    bias = torch.zeros(inp.shape)
    return ops.layer_norm(inp, weight, bias, eps=1e-6)


def qk_norm(q, k, v, rms_q, rms_k):
    return rms_q(q).to(v), rms_k(k).to(v)


def rope(pos: AnyTensor, dim: int, theta: int) -> AnyTensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    # out = out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3], 2, 2)
    out = out.view(out.shape[0], out.shape[1], out.shape[2], 2, 2)
    return out.float()


class MLPEmbedder(ThetaLayer):
    def __init__(self, theta: Theta):
        super().__init__(theta)
        self.in_layer = LinearLayer(theta("in_layer"))
        self.out_layer = LinearLayer(theta("out_layer"))

    def forward(self, x: AnyTensor) -> AnyTensor:
        x = self.in_layer(x)
        x = ops.elementwise(torch.nn.functional.silu, x)
        return self.out_layer(x)


class EmbedND(torch.nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: AnyTensor) -> AnyTensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class LastLayer(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
    ):
        super().__init__(theta)
        self.add_module(
            "adaLN_modulation_linear", LinearLayer(theta("adaLN_modulation.1"))
        )
        self.add_module("linear", LinearLayer(theta("linear")))

    def forward(self, x: AnyTensor, vec: AnyTensor) -> AnyTensor:
        silu = ops.elementwise(F.silu, vec)
        lin = self.adaLN_modulation_linear(silu)
        shift, scale = lin.chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * layer_norm(x) + shift[:, None, :]
        x = self.linear(x)
        return x
