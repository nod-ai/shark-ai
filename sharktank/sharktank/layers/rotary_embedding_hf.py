# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import torch

from .base import BaseLayer

from sharktank.kernels.mlir_kernel import *


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


select_concat = RoPEKernels()


class RotaryEmbeddingHfLayer(BaseLayer):
    """Computes a rotary embedding (RoPE)"""

    def __init__(
        self, *, head_dim: int, rope_theta: float = 10000.0, interleaved: bool = True
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.interleaved = interleaved

    def _compute_theta(self):
        # TODO: Add rope scaling.
        dim = self.head_dim
        # The original paper creates a d/2 dimensional space to represent
        # the polar coordinates.
        #
        # From the original paper:
        #   theta = 10000^{-2 (i - 1) / d}, i \in [1, 2, ..., d/2]
        # which is a convoluted way of saying
        #   theta = (1/base)^{i / d}, i \in range(0, dim, 2)
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def compute_sincos_cache(
        self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute a sin/cos cache based on position_ids. This cache can
        generally be used across. We could also rely on the compiler to do CSE,
        but it can sometimes be really hard to do so.

        position_ids: [bs, seq_len]
        dtype: dtype for the sin/cos cache
        device: device for the sin/cos cache
        output: [bs, seq_len, 1, head_dim // 2], [bs, seq_len, 1, head_dim // 2]
        """
        theta = self._compute_theta()
        theta_expanded = (
            theta[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        ).to(device)
        position_ids_expanded = position_ids[:, None, :].float().to(device)

        freqs = theta_expanded @ position_ids_expanded
        freqs = freqs.transpose(1, 2)

        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        sincos_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the rotary embedding for q/k tensors, given the sin/cos cache.

        q: [bs, seq_len, heads, head_dim]
        k: [bs, seq_len, heads, head_dim]
        sincos_cache: as produced by `compute_sincos_cache`
        output: ([bs, seq_len, heads, head_dim], [bs, seq_len, heads, head_dim])
        """
        assert q.device == k.device
        assert q.dtype == k.dtype

        cos, sin = sincos_cache

        def apply_rotary(x: torch.Tensor):
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

        return apply_rotary(q), apply_rotary(k)
