# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import torch

import amdsharktank.ops as ops
from amdsharktank.types.tensors import AnyTensor, PrimitiveTensor, ReplicatedTensor

from .base import BaseLayer

from amdsharktank.kernels.mlir_kernel import *
import math


def RoPEKernels():
    BS = DynDim.BS
    SL = DynDim.SL
    HEADS = DynDim.HEADS
    HALFDIM = DynDim.HALFDIM
    TWO = StaticDim.TWO(2)

    TY = Dtype.TY

    @mlir_kernel(
        inputs=(
            MLIRTensor[BS, SL, HEADS, HALFDIM, TY],
            MLIRTensor[BS, SL, HEADS, HALFDIM, TY],
        ),
        results=(MLIRTensor[BS, SL, HEADS, TWO, HALFDIM, TY],),
    )
    def rope_select_concat(x1, x2, out=None):
        """
        IREE doesn't have a good concat op yet which can also do fusion. The
        alternatives are tensor.concat or tensor.insert_slice, but both would
        block fusions for RoPE. We use a linalg.generic with arith.select
        on the concat dimension to do the concat instead.
        """

        mlir = """
        !dtype = !x1_dtype
        #trait = {
            indexing_maps = [
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, halfdim)>,
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, halfdim)>,
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, two, halfdim)>
            ],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
        }
        module {
        util.func private @{{kernel_name}}(%x1: !x1,
                                           %x2: !x2) -> !out {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index

            %bs = tensor.dim %x1, %c0 : !x1
            %sl = tensor.dim %x1, %c1 : !x1
            %heads = tensor.dim %x1, %c2 : !x1
            %halfdim = tensor.dim %x1, %c3 : !x1
            %empty = tensor.empty(%bs, %sl, %heads, %halfdim) : !out

            %out = linalg.generic #trait
                   ins(%x1, %x2 : !x1, !x2)
                   outs(%empty: !out) {
                ^bb0(%xs1 : !dtype, %xs2 : !dtype, %o : !dtype):
                %two_dim = linalg.index 3 : index
                // Ideally, when the two dim is unrolled, this condition
                // would become a no-op and we will not do any redundant
                // computation.
                %is_x1 = arith.cmpi eq, %two_dim, %c0 : index
                %val = arith.select %is_x1, %xs1, %xs2 : !dtype
                linalg.yield %val : !dtype
            } -> !out
            util.return %out : !out
        }
        }
        """
        return MLIRSpec(mlir)

    return rope_select_concat


_select_concat = RoPEKernels()


def select_concat(x1: AnyTensor, x2: AnyTensor) -> AnyTensor:
    assert type(x1) == type(x2)
    if isinstance(x1, torch.Tensor):
        return _select_concat(x1, x2)

    if isinstance(x1, PrimitiveTensor):
        return _select_concat(x1.as_torch(), x2.as_torch())

    if isinstance(x1, ReplicatedTensor):
        shards = [select_concat(s1, s2) for s1, s2 in zip(x1.shards, x2.shards)]
        return ReplicatedTensor(ts=shards, devices=x1.devices)

    raise ValueError(f"Unsupported tensor type: {type(x1)}")


class RotaryEmbeddingLayer(BaseLayer):
    """Computes a rotary embedding (RoPE)"""

    def __init__(
        self,
        *,
        head_dim: int,
        rope_theta: float = 10000.0,
        interleaved: bool = True,
        yarn_beta_slow: float | None = None,
        yarn_beta_fast: float | None = None,
        yarn_factor: float | None = None,
        yarn_original_context_len: int | None = None,
        use_base_frequency_scaling: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.interleaved = interleaved

        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_factor = yarn_factor
        self.yarn_original_context_len = yarn_original_context_len
        self.use_base_frequency_scaling = use_base_frequency_scaling

    def _compute_theta(self, device):
        # TODO: Add rope scaling.
        dim = self.head_dim
        # The original paper creates a d/2 dimensional space to represent
        # the polar coordinates.
        #
        # From the original paper:
        #   theta = 10000^{-2 (i - 1) / d}, i \in [1, 2, ..., d/2]
        # which is a convoluted way of saying
        #   theta = (1/base)^{i / d}, i \in range(0, dim, 2)
        if self.use_base_frequency_scaling:
            # gpt-oss base freqs:base^(i/d)
            freqs = self.rope_theta ** (
                ops.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
            )
            # Returning freq and concentration.
            concentration, inv_freqs = self._apply_yarn_base_freq(freqs)
            if not isinstance(concentration, AnyTensor):
                concentration = ops.tensor(
                    concentration, device=device, dtype=torch.float32
                )

        else:
            freqs = 1.0 / (
                self.rope_theta
                ** (ops.arange(0, dim, 2, device=device).to(torch.float32) / dim)
            )
            inv_freqs = self._apply_yarn(freqs)
            concentration = ops.tensor(1.0, device=device, dtype=torch.float32)

        return concentration, inv_freqs

    def _apply_yarn(self, freqs):
        """
        Standard YaRN on inverse frequencies.
        Returns adjusted inverse frequencies.
        """
        yarn_factor = self.yarn_factor
        yarn_beta_slow = self.yarn_beta_slow
        yarn_beta_fast = self.yarn_beta_fast
        yarn_original_context_len = self.yarn_original_context_len
        reqs = [
            yarn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_context_len,
        ]
        any_yarn = any([a is not None for a in reqs])
        use_yarn = all([a is not None for a in reqs])
        assert any_yarn == use_yarn

        if use_yarn:
            low_freq_wavelen = yarn_original_context_len / yarn_beta_slow
            high_freq_wavelen = yarn_original_context_len / yarn_beta_fast

            inv_freq = freqs
            wavelen = 2 * torch.pi / inv_freq
            inv_freq_llama = ops.where(
                wavelen > low_freq_wavelen, inv_freq / yarn_factor, inv_freq
            )

            smooth_factor = (yarn_original_context_len / wavelen - yarn_beta_slow) / (
                yarn_beta_fast - yarn_beta_slow
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / yarn_factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = ops.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return freqs

    def _apply_yarn_base_freq(self, freqs):
        """See YaRN paper: https://arxiv.org/abs/2309.00071
        Base frequency YaRN variant.
        Input:
            freqs_base = rope_theta^(i/d) for i in [0,2,...,d-2]
            Note: This is NOT the inverse frequency.
        Returns:
            concentration: float
                A scalar multiplier for the sin/cos cache
            inv_freq: Tensor[d_half]
                The per-dimension inverse frequencies after applying the gpt-oss YaRN rule.
        Notes:
            - This variant blends between interpolation (1 / (scaling * freqs_base)) and extrapolation (1 / freqs_base)
            across a band of dimensions [low, high], defined in index space from the model's base and context params.
            - If scaling_factor <= 1.0, it defaults to concentration=1.0 and inv_freq = 1 / freqs_base.

        """
        scaling_factor = self.yarn_factor
        rope_ntk_alpha = self.yarn_beta_slow
        rope_ntk_beta = self.yarn_beta_fast
        yarn_original_context_len = self.yarn_original_context_len

        if scaling_factor > 1.0:
            concentration = 0.1 * math.log(scaling_factor) + 1.0
            d_half = self.head_dim // 2
            # NTK by part
            low = (
                d_half
                * math.log(yarn_original_context_len / (rope_ntk_beta * 2 * math.pi))
                / math.log(self.rope_theta)
            )
            high = (
                d_half
                * math.log(yarn_original_context_len / (rope_ntk_alpha * 2 * math.pi))
                / math.log(self.rope_theta)
            )
            assert 0 < low < high < d_half - 1
            interpolation = 1.0 / (scaling_factor * freqs)
            extrapolation = 1.0 / freqs
            ramp = (
                ops.arange(d_half, dtype=torch.float32, device=freqs.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask

        else:
            concentration = 1.0
            inv_freq = 1.0 / freqs

        return concentration, inv_freq

    def compute_sincos_cache(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute a sin/cos cache based on position_ids. This cache can
        generally be used across. We could also rely on the compiler to do CSE,
        but it can sometimes be really hard to do so.

        position_ids: [bs, seq_len]
        dtype: dtype for the sin/cos cache
        device: device for the sin/cos cache
        output: [bs, seq_len, 1, head_dim // 2], [bs, seq_len, 1, head_dim // 2]

        Key intermediates:
            inv_freq: [d_half]
            theta_expanded: [bs, d_half, 1]
            position_ids_expanded: [bs, 1, seq_len]
            angles = theta_expanded @ position_ids_expanded: [bs, d_half, seq_len] -> transpose(1, 2) -> [bs, seq_len, d_half]
        Note:
            - When use_base_frequency_scaling is enabled, a concentration scalar may scale cos/sin.
        """
        concentration, inv_freq = self._compute_theta(device=position_ids.device)
        concentration = ops.reshard_like(concentration, position_ids)
        inv_freq = ops.reshard_like(inv_freq, position_ids)

        # [bs, d_half, 1] x [bs, 1, seq_len] -> [bs, d_half, seq_len] -> [bs, seq_len, d_half]
        theta_expanded = (
            inv_freq[None, :, None]
            .to(torch.float32)
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].to(torch.float32)

        angles = theta_expanded @ position_ids_expanded
        angles = angles.transpose(1, 2)

        cos = (angles.cos() * concentration).to(dtype)
        sin = (angles.sin() * concentration).to(dtype)

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return cos, sin

    def forward(
        self,
        q: AnyTensor,
        sincos_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[AnyTensor, AnyTensor]:
        """
        Compute the rotary embedding for q/k tensors, given the sin/cos cache.

        q: [bs, seq_len, heads, head_dim]
        sincos_cache: as produced by `compute_sincos_cache`
        output: ([bs, seq_len, heads, head_dim], [bs, seq_len, heads, head_dim])

        Notes: self.interleaved = False when working with base frequency scaling
        """

        cos, sin = sincos_cache
        dtype = cos.dtype
        cos = cos.to(device=q.device)
        sin = sin.to(device=q.device)

        def apply_rotary(x: AnyTensor):
            # The original RoPE paper forms "interleaved" pairs along the head
            # dimension, i.e. it forms pairs like:
            #   [0, 1, 2, 3, 4, 5...] -> [(0, 1), (2, 3), (4, 5), ...]
            # and computes the embedding as:
            #   [rope_0(0, 1), rope_1(0, 1), rope_0(2, 3), rope_1(2, 3)]
            #
            # This is somewhat problemetic, as this is effectively creating
            # a innermost dimension of 2, which we are working over.
            #
            # However, it doesn't really matter how the pairs are formed,
            # as long as training and inference form the same pairs. HuggingFace
            # uses an alternative "interweaved" implementation, where pairs are
            # formed as:
            #   [0, 1, 2 ... 64, 65, 66...] -> [(0, 64), (1, 65), (2, 66), ...]
            # and computes the embedding as
            #   [rope_0(0, 64), rope_0(1, 65) ... rope_1(0, 64), rope_1(1, 65)]
            #
            # This implementation is prefered, because it creates an outer
            # dimension of 2, which we can handle much better in general.
            #
            # Another interesting thing to note about RoPE is the output of
            # RoPE can use either "interleaved" or "interweaved" pair layout,
            # as long as both Q/K use the same implementation, since the
            # computation involved is rope(Q) @ rope(K).T, where the head_dim
            # is reduced (permutations along reduction dimensions are
            # preserved).
            #
            # With this information, our implementation of RoPE supports taking
            # both interleaved/interweaved pairing as input, however the
            # interweaved pairing as input is prefered. If your model uses
            # interleaved pairing, this can be converted to interweaved pairing
            # by transposing the weights like hugging face does (hugging face
            # uses interweaved pairing):
            # https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
            #
            # Our implementation only produces interweaved output. Note that
            # HuggingFace also always produces the interweaved output.
            # NOTE: Any elementwise operations along the reduction dimension
            # post rope need to also be permuted the same way (this applies to
            # testing equivalence checks too).

            if self.interleaved:
                # TODO: Currently, we do a strided extract slice here. This, in
                # codegen would turn into a slow_memcpy dispatch to do this
                # slicing. There are two ways to avoid this:
                #   1. Use gather and fuse the gather.
                #   2. Do expand_shape and then extract.
                #   3. Add a deinterleave operation and fuse it.
                #
                # Possible issues with each of these (respectively):
                #   1. Using gathers on the innermost dimension is effectively
                #   taking contiguity information away. This ruins the
                #   possibility of getting any good loads.
                #   2. The expand_shape on the innermost dimension creates an
                #   innermost dimension of 2. It is not the nicest to work with
                #   when targeting intrinsics which need a dim of atleast 16.
                #   3. We do not have a deinterleave operation in IREE as of
                #   today.
                #
                # For now, we live with a slow_memcpy until someone complains.
                x_real = x[..., 0 : self.head_dim : 2]
                x_imag = x[..., 1 : self.head_dim : 2]
            else:
                x_real = x[..., : self.head_dim // 2]
                x_imag = x[..., self.head_dim // 2 :]

            x1 = x_real * cos - x_imag * sin
            x2 = x_imag * cos + x_real * sin

            cated = select_concat(x1, x2)
            # Collapse the last two dimensions.
            cated = cated.flatten(start_dim=-2, end_dim=-1)

            return cated

        return apply_rotary(q.to(dtype)).to(q.dtype)
