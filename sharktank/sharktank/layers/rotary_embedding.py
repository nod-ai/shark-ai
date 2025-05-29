# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import torch

from .base import BaseLayer
from sharktank import ops, kernels
from sharktank.types import (
    SplitPrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    unbox_tensor,
)
from sharktank.kernels.mlir_kernel import *


def RoPEApplyRotaryEmbed():
    BS = DynDim.BS
    SL = DynDim.SL
    HEADS = StaticDim.HEADS
    HALFDIM = StaticDim.HALFDIM
    TWO = StaticDim.TWO(2)

    X_TY = Dtype.X_TY
    # The theta is always calculated in float32 to preserve numerics.
    F_TY = Dtype.F_TY(torch.float32)

    half_shape = MLIRTensor[BS, SL, HEADS, HALFDIM, X_TY]
    freq_shape = MLIRTensor[BS, SL, HALFDIM, F_TY]
    out_shape = MLIRTensor[BS, SL, HEADS, TWO, HALFDIM, X_TY]

    @mlir_kernel(
        inputs=(
            half_shape,
            half_shape,
            freq_shape,
        ),
        results=(out_shape,),
    )
    def apply_rope(real, imag, freq, out=None):
        mlir = """
        !x_dtype = !real_dtype
        !f_dtype = !freq_dtype

        #trait = {
            indexing_maps = [
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, halfdim)>,
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, halfdim)>,
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, halfdim)>,
                affine_map<(bs, sl, heads, two, halfdim) -> (bs, sl, heads, two, halfdim)>
            ],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
        }

        module {
        util.func private @{{kernel_name}}(%real: !real,
                                           %imag: !imag,
                                           %freq: !freq) -> !out {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index

            %bs = tensor.dim %real, %c0 : !real
            %sl = tensor.dim %real, %c1 : !real

            %empty = tensor.empty(%bs, %sl) : !out
            %out = linalg.generic
                   #trait
                   ins(%real, %imag, %freq : !real, !imag, !freq)
                   outs(%empty: !out) {
                ^bb0(%r : !x_dtype, %i : !x_dtype, %f : !f_dtype, %o : !x_dtype):
                // kernel libraries use __sincosf to compute sin and cos
                // at the same time, maybe something worth looking at.
                {% if X_TY != F_TY %}
                    %cosf32 = math.cos %f : !f_dtype
                    %sinf32 = math.sin %f : !f_dtype
                    // the trig functions are calculated in float32, but the input
                    // is usually a narrower dtype like f16/bf16.
                    %cos = arith.truncf %cosf32 : !f_dtype to !x_dtype
                    %sin = arith.truncf %sinf32 : !f_dtype to !x_dtype
                {% else %}
                    %cos = math.cos %f : !x_dtype
                    %sin = math.sin %f : !x_dtype
                {% endif %}
                // x1 = real * cos - imag * sin
                %realcos = arith.mulf %r, %cos : !x_dtype
                %imagsin = arith.mulf %i, %sin : !x_dtype
                %x1 = arith.subf %realcos, %imagsin : !x_dtype
                // x2 = imag * cos + real * sin
                %imagcos = arith.mulf %i, %cos : !x_dtype
                %realsin = arith.mulf %r, %sin : !x_dtype
                %x2 = arith.addf %imagcos, %realsin : !x_dtype
                // select based on the `two` dimension.
                %two_dim = linalg.index 3 : index
                // Ideally, when the two dim is unrolled, this condition
                // would become a no-op and we will not do any redundant
                // computation.
                %is_x1 = arith.cmpi eq, %two_dim, %c0 : index
                %val = arith.select %is_x1, %x1, %x2 : !x_dtype
                linalg.yield %val : !x_dtype
            } -> !out

            util.return %out : !out
        }
        }
        """
        return MLIRSpec(mlir)

    return apply_rope


apply_rotary_embedding_kernel = RoPEApplyRotaryEmbed()


class RotaryEmbeddingLayer(BaseLayer):
    """Computes a rotary embedding in the style popularized by llama (RoPE)."""

    def __init__(
        self,
        *,
        rope_dimension_count: int,
        max_seqlen: int,
        rope_freq_base: Optional[float],
        device: Optional[torch.device] = None,
        use_hf: bool = False,
        rope_scaling_type: Optional[str] = None,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism: bool = False,
        dtype: torch.dtype = torch.float32,
        devices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.device = device
        self.rope_dimension_count = rope_dimension_count
        self.max_seqlen = max_seqlen
        self.use_hf = use_hf
        self.rope_scaling_type = rope_scaling_type
        self.dtype = dtype
        self.rope_freq_base = rope_freq_base if rope_freq_base is not None else 10000.0
        self.tensor_parallelism_size = tensor_parallelism_size
        self.pipeline_parallelism = pipeline_parallelism
        self.devices = (
            devices
            if devices is not None
            else tuple(range(self.tensor_parallelism_size))
        )
        self.theta = self._calculate_theta()

    def _calculate_theta(self) -> torch.Tensor:
        dim = self.rope_dimension_count
        # The original paper creates a d/2 dimensional space to represent
        # the polar coordinates.
        #
        # From the original paper:
        #   theta = 10000^{-2 (i - 1) / d}, i \in [1, 2, ..., d/2]
        # which is a convoluted way of saying
        #   theta = (1/base)^{i / d}, i \in range(0, dim, 2)
        rope_rcp_theta = 1.0 / self.rope_freq_base
        freqs = rope_rcp_theta ** (torch.arange(0, dim, 2).float() / dim)

        if self.rope_scaling_type == "llama3":
            # llama3.1 introduced rope scaling to normalize the theta better.
            # The theory is based on the original llama3.1 implementation:
            # https://github.com/meta-llama/llama-models/blob/709a61fd810157f75fbb314e7287089eec06d9c3/models/llama3_1/api/model.py#L41
            # TODO: Not all models use rope scaling, fix this.
            # TODO: get these values from Dataset. Current values are derived
            # from llama3.1 reference link above.
            rope_factor = 8
            low_freq_factor = 1
            high_freq_factor = 4
            old_context_len = 8192

            # The reference implementation is based on flash-infer. This
            # implementation uses clamping instead of conditionals which is
            # much better for a tensor compiler:
            # https://github.com/flashinfer-ai/flashinfer/commit/4c89decadc8ae9f261cae97c350064156e66bc09#diff-e797f0f37e32a5e08c50ef190459c873fcb33ef6334333cef1e4e2d931308fa3
            smooth_a = old_context_len / (
                2 * torch.pi * (high_freq_factor - low_freq_factor)
            )
            smooth_b = -1.0 / (high_freq_factor / low_freq_factor - 1.0)

            rope_rcp_scale = 1.0 / rope_factor

            smooth = freqs * smooth_a + smooth_b
            # Clamp to [0, 1]
            smooth = torch.clamp(smooth, 0.0, 1.0)
            freqs = (1 - smooth) * (freqs * rope_rcp_scale) + smooth * freqs
        elif self.rope_scaling_type is not None:
            raise ValueError(f"{self.rope_scaling_type} NYI")

        return freqs

    @property
    def rotary_embed_table(self):
        return self._create_rotary_embed_table()

    def forward(
        self,
        *,
        xt: Union[torch.Tensor, ShardedTensor],
        start_index: int,
    ):
        table = self.rotary_embed_table
        if isinstance(xt, ReplicatedTensor):
            return ReplicatedTensor(
                ts=[
                    self.forward_unsharded(
                        xt=unbox_tensor(s),
                        start_index=start_index,
                        rotary_embed_table=unbox_tensor(t),
                    )
                    for s, t in zip(xt.shards, table.shards)
                ],
                devices=table.devices,
            )

        if not isinstance(xt, ShardedTensor):
            return self.forward_unsharded(
                xt=xt,
                start_index=start_index,
                rotary_embed_table=table,
            )

        if isinstance(xt, SplitPrimitiveTensor):
            assert (
                not self.use_hf or xt.shard_dim == len(xt.shape) - 1
            ), "We rotate the last dim in that case causing awkwardness, so sharding it is disallowed"
            assert (
                isinstance(table, ShardedTensor) and xt.shard_count == table.shard_count
            )
            rotary_shards = [unbox_tensor(shard) for shard in table.shards]

            xt_shards = [
                self.forward_unsharded(
                    xt=unbox_tensor(xt_shard),
                    start_index=start_index,
                    rotary_embed_table=rotary_shard,
                )
                for xt_shard, rotary_shard in zip(xt.shards, rotary_shards)
            ]
            return xt.clone(ts=xt_shards)

        raise NotImplementedError(
            f"Rotary embedding layer not implemented for input tensor type {type(xt)}"
        )

    def forward_unsharded(
        self,
        *,
        xt: torch.Tensor,
        start_index: int,
        rotary_embed_table: Optional[torch.Tensor],
    ):
        # freqs_cis shape: max_sl, dim
        # xq_, xk_ shape: bs, sl, heads, dim
        bs, sl, heads, dim = xt.shape

        if self.use_hf:
            freqs_cis = rotary_embed_table
            # Slice from max to current sequence length
            cos, sin = [x[start_index : start_index + sl, :] for x in freqs_cis]
            # expand to 1, sl, 1, dim and repeat per bs
            cos = cos[None, :, None, :].repeat(xt.shape[0], 1, 1, 1)
            sin = sin[None, :, None, :].repeat(xt.shape[0], 1, 1, 1)
            xt_real = xt[..., : dim // 2]
            xt_imag = xt[..., dim // 2 :]
            x1 = xt_real * cos - xt_imag * sin
            x2 = xt_imag * cos + xt_real * sin
            return torch.cat((x1, x2), dim=-1)

        # Offset the table based on starting position.
        freqs_cis = torch.arange(sl, device=xt.device) + start_index
        freqs_cis = self._compute_rotary_embed_table(freqs_cis)

        assert (
            freqs_cis.shape[0] >= sl
        ), f"Sequence length longer than embedding table ({sl} vs {freqs_cis.shape[0]})"

        freqs_cis = ops.repeat(freqs_cis[None, :, :], (xt.shape[0], 1, 1))

        xt_real = xt[..., : dim // 2]
        xt_imag = xt[..., dim // 2 :]
        xt_out: torch.Tensor = apply_rotary_embedding_kernel(
            xt_real, xt_imag, freqs_cis
        )
        xt_out = xt_out.reshape(*xt.shape)

        return ops.to(xt_out, xt.dtype)

    def compute_batch_mask(
        self, start_positions: Union[torch.Tensor, ReplicatedTensor], batch_seq_len: int
    ) -> torch.Tensor:
        # TODO: I'm pretty sure this function is only correct because batch_seq_len is always 1
        """Computes a mask for a batch that can be repeatedly applied.

        Args:
          start_positions: Tensor of [bs] with start positions for every sequence
            in the batch.
          batch_seq_len: The sequence length dimension of the batch.
        Returns:
          Tensor of [bs, sl, 1, d] that will be later passed to apply_batch_mask.
        """
        self.trace_tensor("rope.start_positions", start_positions)
        positions_seq = torch.arange(0, batch_seq_len, device=self.device).unsqueeze(
            0
        ) + start_positions.unsqueeze(1)
        # Broadcast lookup to [b, ...].
        self.trace_tensor("rope.positions_seq", positions_seq)
        if self.use_hf:
            freqs_cis = self.rotary_embed_table
            cos, sin = [x[positions_seq.flatten(), :] for x in freqs_cis]
            freqs_cis = (cos[:, None, None, :], sin[:, None, None, :])
            return freqs_cis

        shape = positions_seq.shape
        if isinstance(positions_seq, ReplicatedTensor):
            ts = [
                self._compute_rotary_embed_table(s.flatten()).unflatten(0, shape)
                for s in positions_seq.shards
            ]
            freqs_cis = ReplicatedTensor(ts=ts)
        else:
            freqs_cis = self._compute_rotary_embed_table(positions_seq.flatten())

        return freqs_cis.unsqueeze(1)

    def apply_batched_mask(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor, ReplicatedTensor],
        mask: Union[torch.Tensor, ReplicatedTensor],
    ) -> Union[SplitPrimitiveTensor, ReplicatedTensor]:
        if not isinstance(xt, ShardedTensor):
            return self.apply_batched_mask_unsharded(xt=xt, mask=mask)

        assert isinstance(mask, ReplicatedTensor) and mask.shard_count == xt.shard_count
        xt_shards = [
            self.apply_batched_mask_unsharded(
                xt=unbox_tensor(xt_shard),
                mask=unbox_tensor(mask_shard),
            )
            for xt_shard, mask_shard in zip(xt.shards, mask.shards)
        ]
        return xt.clone(ts=xt_shards)

    def apply_batched_mask_unsharded(self, *, xt: torch.Tensor, mask: torch.Tensor):
        """Applies the embedding to a ragged batch of queries and keys.

        This does a more complicated indexing operation for cases when the each
        sequence in the batch has a potentially different start position.

        positions should be of [bs, sl] and enumerate positions of all tokens.
        """
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim

        _, _, _, dim = xt.shape

        if self.use_hf:
            cos, sin = mask
            xt_real = xt[..., : dim // 2]
            xt_imag = xt[..., dim // 2 :]
            x1 = xt_real * cos - xt_imag * sin
            x2 = xt_imag * cos + xt_real * sin
            return torch.cat((x1, x2), dim=-1)

        xt_real = xt[..., : dim // 2]
        xt_imag = xt[..., dim // 2 :]
        xt_out: torch.Tensor = apply_rotary_embedding_kernel(xt_real, xt_imag, mask)
        xt_out = xt_out.reshape(*xt.shape)
        return xt_out

    def _compute_rotary_embed_table(self, t):
        emb = torch.outer(t, self.theta)
        if not self.use_hf:
            return emb

        # Note that the freqs and trig functions are calculated in float32, and
        # then downcasted to the input type. Without this, there is a big
        # accuracy loss. The mlir_kernel also does the same thing, where the
        # trig functions are calculated in float32, but then truncated to
        # the input type.
        cos = torch.cos(emb).to(self.dtype)
        sin = torch.sin(emb).to(self.dtype)
        return (cos, sin)

    def _create_rotary_embed_table(self):
        t = torch.arange(self.max_seqlen, device=self.device)
        freqs_cis = self._compute_rotary_embed_table(t)
        return self._replicate(freqs_cis)

    def _replicate(self, t):
        if self.tensor_parallelism_size > 1 or self.pipeline_parallelism:
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t = ops.replicate(t, self.tensor_parallelism_size, devices=self.devices)

        return t
