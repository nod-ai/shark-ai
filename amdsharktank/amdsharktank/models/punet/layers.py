# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn

from amdsharktank import ops
from amdsharktank.layers import *
from amdsharktank.types import *
from .config import *


__all__ = [
    "ACTIVATION_FUNCTIONS",
    "CrossAttnUpDownBlock2D",
    "GroupNormLayer",
    "TimestepEmbedding",
    "TimestepProjection",
    "UpDownBlock2D",
]

################################################################################
# Down blocks.
################################################################################


class UpDownBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        num_layers: int,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        resnet_out_scale_factor: Optional[float],
        resnet_time_scale_shift: str,
        temb_channels: int,
        dropout: float,
        add_upsample: bool = False,
        add_downsample: bool = False,
        downsample_padding: int = 0,
    ):
        super().__init__(theta)
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    theta("resnets", i),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=resnet_out_scale_factor,
                    time_embedding_norm=resnet_time_scale_shift,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(theta("downsamplers", "0"), padding=downsample_padding)]
            )
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(theta("upsamplers", 0), padding=1)]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        *,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        upsample_size: Optional[int] = None,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...] = None,
    ):
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            # Up blocks will have resnet residuals from the down phase. If present,
            # pop each one and cat.
            if res_hidden_states_tuple is not None:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = ops.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpDownBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        num_layers: int,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        resnet_out_scale_factor: Optional[float],
        resnet_time_scale_shift: str,
        temb_channels: int,
        dropout: float,
        num_attention_heads: int,
        transformer_layers_per_block: Sequence[int],
        cross_attention_dim: int,
        use_linear_projection: bool,
        num_prefix_resnets: int = 0,
        add_upsample: bool = False,
        add_downsample: bool = False,
        downsample_padding: int = 0,
    ):
        super().__init__(theta)
        resnets = []
        attentions = []

        transformer_counts = (
            [transformer_layers_per_block] * num_layers
            if isinstance(transformer_layers_per_block, int)
            else transformer_layers_per_block
        )
        for i in range(num_prefix_resnets + num_layers):
            resnets.append(
                ResnetBlock2D(
                    theta("resnets", i),
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=resnet_out_scale_factor,
                )
            )

        for i in range(num_layers):
            attentions.append(
                ContinuousInputTransformer2DModel(
                    theta("attentions", i),
                    num_attention_heads=num_attention_heads,
                    num_layers=transformer_counts[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(theta("downsamplers", 0), padding=downsample_padding)]
            )
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(theta("upsamplers", 0), padding=1)]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        *,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        upsample_size: Optional[int] = None,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...] = None,
    ):
        output_states = ()
        layer_count = max(len(self.resnets), len(self.attentions))
        assert len(self.resnets) >= len(self.attentions)
        for i in range(layer_count):
            # Up blocks will have resnet residuals from the down phase. If present,
            # pop each one and cat.
            if res_hidden_states_tuple is not None:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = ops.cat([hidden_states, res_hidden_states], dim=1)

            # Alternate resnet, attention layers, allowing attention to be short.
            hidden_states = self.resnets[i](hidden_states, temb)
            if i < len(self.attentions):
                hidden_states = self.attentions[i](
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


################################################################################
# Downsample/Upsample blocks.
################################################################################


class Downsample2D(ThetaLayer):
    def __init__(self, theta: Theta, padding: int):
        super().__init__(theta)
        assert padding != 0
        self.conv = Conv2DLayer(theta("conv"), padding=[padding, padding], stride=2)

    def forward(self, hidden_states: torch.Tensor):
        return self.conv(hidden_states)


class Upsample2D(ThetaLayer):
    def __init__(self, theta: Theta, padding: int, interpolate: bool = True):
        super().__init__(theta)
        assert padding != 0
        self.interpolate = interpolate
        self.conv = Conv2DLayer(theta("conv"), padding=[padding, padding])

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None):
        if self.interpolate:
            if output_size is None:
                hidden_states = ops.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = ops.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )
        hidden_states = self.conv(hidden_states)
        return hidden_states


################################################################################
# Transformer block.
################################################################################


class ContinuousInputTransformer2DModel(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        num_attention_heads,
        num_layers,
        cross_attention_dim,
        norm_num_groups,
        use_linear_projection,
    ):
        super().__init__(theta)
        assert use_linear_projection, "NYI: not use_linear_projection"
        self.norm = GroupNormLayer(theta("norm"), num_groups=norm_num_groups, eps=1e-6)
        self.proj_in = LinearLayer(theta("proj_in"))
        self.proj_out = LinearLayer(theta("proj_out"))
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    theta("transformer_blocks", i),
                    num_attention_heads=num_attention_heads,
                    # attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    ):
        assert attention_mask is None, "NYI: attention_mask != None"
        assert encoder_attention_mask is None, "NYI: encoder_attention_mask != None"

        # Project input.
        bs, inner_dim, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            bs, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks.
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Output.
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(bs, height, width, inner_dim).permute(
            0, 3, 1, 2
        )
        output = hidden_states + residual
        return output


class BasicTransformerBlock(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        num_attention_heads: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__(theta)
        self.norm1 = LayerNormLayer(theta("norm1"), eps=norm_eps)
        self.norm2 = LayerNormLayer(theta("norm2"), eps=norm_eps)
        self.norm3 = LayerNormLayer(theta("norm3"), eps=norm_eps)

        self.attn1 = AttentionLayer(theta("attn1"), heads=num_attention_heads)
        self.attn2 = AttentionLayer(theta("attn2"), heads=num_attention_heads)

        # Diffusers BasicTransformerBlock has an unfortunate structure for
        # the FF net layers:
        #   ff.net.0 = GEGLU
        #   ff.net.1 = Dropout (unused here)
        #   ff.net.2 = Linear
        self.ff_proj_in = GEGLULayer(theta("ff", "net", 0))
        self.ff_proj_out = LinearLayer(theta("ff", "net", 2))

    def forward(
        self,
        hidden_states,
        *,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
    ):
        # Attention 1.
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # Attention 2.
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # Feed forward.
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff_proj_in(norm_hidden_states)
        ff_output = self.ff_proj_out(ff_output)
        hidden_states = ff_output + hidden_states
        return hidden_states


class AttentionLayer(ThetaLayer):
    def __init__(self, theta: Theta, *, heads: int):
        super().__init__(theta)
        self.heads = heads
        # Diffusers models this wonky. They have dropout as the second layer of
        # to_out. But ignored for inference.
        self.to_out = LinearLayer(theta("to_out", 0))

    def _reshape_qkv(self, t):
        bs = t.shape[0]
        inner_dim = t.shape[-1]
        head_dim = inner_dim // self.heads

        if isinstance(t, PlanarQuantizedTensor) and isinstance(
            t.layout, TensorScaledLayout
        ):
            layout = t.layout.view(bs, -1, self.heads, head_dim)
            return PlanarQuantizedTensor(shape=layout.shape, layout=layout).transpose(
                1, 2
            )

        return t.view(bs, -1, self.heads, head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        bs, *_ = hidden_states.shape
        query, key, value = self.project_qkv(hidden_states, encoder_hidden_states)

        out_q = self.theta.optional_tensor("out_q")
        out_k = self.theta.optional_tensor("out_k")
        out_v = self.theta.optional_tensor("out_v")

        query = query if out_q is None else ops.quantize(query, out_q)
        key = key if out_k is None else ops.quantize(key, out_k)
        value = value if out_v is None else ops.quantize(value, out_v)

        inner_dim = key.shape[-1]

        query = self._reshape_qkv(query)
        key = self._reshape_qkv(key)
        value = self._reshape_qkv(value)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = ops.scaled_dot_product_attention(
            query, key, value, a=attention_mask, is_causal=False, scale=None
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, inner_dim)

        # Linear proj.
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def project_qkv(self, hidden_states, encoder_hidden_states):
        # There are several ways to do the qkv projection, depending on how/if
        # activation quantization is fused:
        #  * MHA QKV fusion: All projections share activation quant..
        #  * MCHA KV fusion: Q is quantized standalone, and KV share activation quant.
        #  * Unfused: Q/K/V activations are quantized individually.
        theta = self.theta
        kv_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        if "to_qkv" in theta:
            # Fused qkv quantization. Sub-layers of:
            #  to_qkv
            #  to_qkv.to_q
            #  to_qkv.to_k
            #  to_qkv.to_v
            assert (
                encoder_hidden_states is None
            ), "QKV quant fusion incompatible with MCHA encoder_hidden_states"
            to_qkv_theta = theta("to_qkv")
            to_q_theta = to_qkv_theta("to_q")
            to_k_theta = to_qkv_theta("to_k")
            to_v_theta = to_qkv_theta("to_v")
            qkv_activation = self._quantize_activation(to_qkv_theta, hidden_states)
            q = self._apply_linear(to_q_theta, qkv_activation)
            k = self._apply_linear(to_k_theta, qkv_activation)
            v = self._apply_linear(to_v_theta, qkv_activation)

            return (q, k, v)

        elif "to_q" in theta and "to_kv" in theta:
            # Unfused q, fused kv quantization. Sub-layers of:
            #  to_q
            #  to_kv.to_k
            #  to_kv.to_v
            to_q_theta = theta("to_q")
            to_kv_theta = theta("to_kv")
            to_k_theta = to_kv_theta("to_k")
            to_v_theta = to_kv_theta("to_v")
            q_activation = self._quantize_activation(to_q_theta, hidden_states)
            kv_activation = self._quantize_activation(to_kv_theta, kv_states)
            q = self._apply_linear(to_q_theta, q_activation)
            k = self._apply_linear(to_k_theta, kv_activation)
            v = self._apply_linear(to_v_theta, kv_activation)

            return q, k, v
        else:
            # Unfused.
            to_q_theta = theta("to_q")
            to_k_theta = theta("to_k")
            to_v_theta = theta("to_v")
            return (
                self._apply_linear(
                    to_q_theta, self._quantize_activation(to_q_theta, hidden_states)
                ),
                self._apply_linear(
                    to_k_theta, self._quantize_activation(to_k_theta, kv_states)
                ),
                self._apply_linear(
                    to_v_theta, self._quantize_activation(to_v_theta, kv_states)
                ),
            )

    def _quantize_activation(self, act_theta: Theta, x):
        # Matches the input conditioning sequence from LinearLayer.
        premul_input = act_theta.optional_tensor("premul_input")
        q_input = act_theta.optional_tensor("q_input")
        if premul_input is not None:
            x = ops.elementwise(torch.mul, x, premul_input)
        if q_input is not None:
            x = ops.quantize(x, q_input)
        return x

    def _apply_linear(self, weight_theta: Theta, x):
        # Matches the primary computation (minus activation conditioning)
        # of LinearLayer.
        weight = weight_theta.tensor("weight")
        bias = weight_theta.optional_tensor("bias")
        y = ops.linear(x, weight, bias)
        if isinstance(y, QuantizedTensor):
            y = y.unpack().dequant()
        return y


################################################################################
# Resnet block.
################################################################################


class ResnetBlock2D(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        groups: int,
        eps: float,
        non_linearity: str,
        output_scale_factor: Optional[float],
        dropout: float,
        temb_channels: int,
        time_embedding_norm: str,
    ):
        super().__init__(theta)
        assert dropout == 0.0, "ResnetBlock2D currently does not support dropout"
        self.norm1 = GroupNormLayer(theta("norm1"), num_groups=groups, eps=eps)
        self.conv1 = Conv2DLayer(theta("conv1"), padding=(1, 1))
        self.norm2 = GroupNormLayer(theta("norm2"), num_groups=groups, eps=eps)
        self.conv2 = Conv2DLayer(theta("conv2"), padding=(1, 1))
        self.nonlinearity = ACTIVATION_FUNCTIONS[non_linearity]
        self.output_scale_factor = output_scale_factor

        self.time_emb_proj = None
        if temb_channels is not None:
            assert (
                time_embedding_norm == "default"
            ), f"NYI: ResnetBlock2D(time_embedding_norm={time_embedding_norm})"
            self.time_emb_proj = LinearLayer(theta("time_emb_proj"))

        self.conv_shortcut = None
        if "conv_shortcut" in theta.keys:
            self.conv_shortcut = Conv2DLayer(theta("conv_shortcut"), padding=(0, 0))

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = ops.elementwise(self.nonlinearity, hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = ops.elementwise(self.nonlinearity, temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            hidden_states = ops.elementwise(torch.add, hidden_states, temb)

        hidden_states = self.norm2(hidden_states)
        hidden_states = ops.elementwise(self.nonlinearity, hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states
        if self.output_scale_factor is not None:
            output_tensor = output_tensor / self.output_scale_factor

        return output_tensor


################################################################################
# Utility layers.
################################################################################

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


class TimestepEmbedding(ThetaLayer):
    """Computes the embedding of projected timesteps.

    This consists of two linear layers with activation applied between.
    """

    def __init__(
        self, theta: Theta, in_channels: int, time_embed_dim: int, act_fn: str
    ):
        super().__init__(theta)
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        try:
            self.act_fn = ACTIVATION_FUNCTIONS[act_fn]
        except KeyError as e:
            raise AssertionError(f"Unknown activation function '{act_fn}'") from e
        self.linear_1 = LinearLayer(theta("linear_1"))
        self.linear_2 = LinearLayer(theta("linear_2"))

    def forward(self, sample):
        h = sample
        h = self.linear_1(h)
        h = ops.elementwise(self.act_fn, h)
        h = self.linear_2(h)
        return h


class TimestepProjection(nn.Module):
    """Adapted from diffusers embeddings.get_timestep_embedding(), which claims:
        'This matches the implementation in Denoising Diffusion Probabilistic Models:
        Create sinusoidal timestep embeddings.'

    Args:
      embedding_dim: the dimension of the output.
      max_period: controls the minimum frequency of the
                  embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_period: int = 10000,
        downscale_freq_shift: float = 1.0,
        scale: float = 1.0,
        flip_sin_to_cos: bool = False,
    ):
        super().__init__()
        self.downscale_freq_shift = downscale_freq_shift
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.scale = scale
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, timesteps):
        """Args:
          timesteps: a 1-D Tensor of N indices, one per batch element.
                     These may be fractional.
        Returns:
          An [N x dim] Tensor of positional embeddings.
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.embedding_dim
        max_period = self.max_period
        downscale_freq_shift = self.downscale_freq_shift

        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].to(torch.float32) * emb[None, :]

        # scale embeddings
        emb = self.scale * emb

        # concat sine and cosine embeddings
        emb = ops.cat([ops.sin(emb), ops.cos(emb)], dim=-1)

        # flip sin and cos embeddings
        if self.flip_sin_to_cos:
            emb = ops.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            emb = ops.pad(emb, (0, 1, 0, 0))
        return emb


################################################################################
# Layers that need to be made common once stable.
################################################################################


class GroupNormLayer(ThetaLayer):
    def __init__(self, theta: Theta, num_groups: int, eps: float, affine: bool = True):
        super().__init__(theta)
        assert affine, "NYI: GroupNormLayer(affine=False)"
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.theta.tensor("weight")
        bias = self.theta.tensor("bias")
        return ops.group_norm_affine(
            input, weight, bias, num_groups=self.num_groups, eps=self.eps
        )


class LayerNormLayer(ThetaLayer):
    def __init__(self, theta: Theta, eps: float):
        super().__init__(theta)
        self.eps = eps
        self.weight = theta.tensor("weight")
        self.bias = None
        if "bias" in theta.keys:
            self.bias = theta.tensor("bias")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ops.layer_norm(
            input,
            self.weight,
            self.bias,
            eps=self.eps,
        )


class GEGLULayer(ThetaLayer):
    def __init__(self, theta: Theta):
        super().__init__(theta)
        self.proj = LinearLayer(theta("proj"))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * nn.functional.gelu(gate)
