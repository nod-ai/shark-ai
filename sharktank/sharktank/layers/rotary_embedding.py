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
    ShardedTensor,
    SplitPrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    unbox_tensor,
)


def build_rotary_layer(
    tensor_parallelism_size: int = 1,
    pipeline_parallelism: bool = False,
    devices=None,
    **kwargs,
):
    rotary_layer = RotaryEmbeddingLayer(**kwargs)
    return ShardedRotaryLayer(
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism=pipeline_parallelism,
        rotary_layer=rotary_layer,
        devices=devices,
    )


# We wrap a shardless rotary layer so the sharding behavior can be handled independently of the numerics.
class ShardedRotaryLayer(BaseLayer):
    def __init__(
        self,
        *,
        tensor_parallelism_size: int,
        pipeline_parallelism: bool,
        rotary_layer,
        devices,
    ):
        super().__init__()
        self._pipeline_parallelism = pipeline_parallelism
        self._tensor_parallelism_size = tensor_parallelism_size
        self._rotary_layer = rotary_layer
        self._devices = (
            devices
            if devices is not None
            else tuple(range(self._tensor_parallelism_size))
        )

    def rotary_embed_table(self):
        t = self._rotary_layer.create_rotary_embed_table()
        if self._tensor_parallelism_size > 1 or self._pipeline_parallelism:
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t = ops.replicate(t, self._tensor_parallelism_size, devices=self._devices)

        return t

    def forward(
        self,
        *,
        xt: Union[torch.Tensor, ShardedTensor],
        start_index: int,
    ):
        table = self.rotary_embed_table()

        if not isinstance(table, ShardedTensor):
            return self._rotary_layer(
                xt=xt, start_index=start_index, rotary_embed_table=table
            )

        shards = [
            self._rotary_layer(xt=xs, start_index=start_index, rotary_embed_table=ts)
            for xs, ts in zip(xt.shards, table.shards)
        ]
        return xt.clone(ts=shards)

    def compute_batch_mask(
        self, start_positions: Union[torch.Tensor, ShardedTensor], batch_seq_len: int
    ) -> torch.Tensor:
        if isinstance(start_positions, ShardedTensor):
            shards = []
            table = self.rotary_embed_table()
            for s, ts in zip(start_positions.shards, table.shards):
                shard = self._rotary_layer.compute_batch_mask(
                    start_positions=s,
                    batch_seq_len=batch_seq_len,
                    rotary_embed_table=ts,
                )
                shards.append(shard)

            return start_positions.clone(ts=shards)

        return self._rotary_layer.compute_batch_mask(
            start_positions=start_positions,
            batch_seq_len=batch_seq_len,
            rotary_embed_table=self.rotary_embed_table(),
        )

    def apply_batched_mask(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor, ReplicatedTensor],
        mask: Union[torch.Tensor, ReplicatedTensor],
    ) -> Union[SplitPrimitiveTensor, ReplicatedTensor]:

        if not isinstance(xt, ShardedTensor):
            return self._rotary_layer.apply_batched_mask(xt=xt, mask=mask)

        assert isinstance(mask, ReplicatedTensor) and mask.shard_count == xt.shard_count
        xt_shards = [
            self._rotary_layer.apply_batched_mask(
                xt=unbox_tensor(xt_shard),
                mask=unbox_tensor(mask_shard),
            )
            for xt_shard, mask_shard in zip(xt.shards, mask.shards)
        ]
        return xt.clone(ts=xt_shards)


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
        use_table: bool = True,
        dtype: torch.dtype = torch.float32,
        yarn_beta_slow: float | None = None,
        yarn_beta_fast: float | None = None,
        yarn_factor: float | None = None,
        yarn_original_context_len: int | None = None,
        model_arch: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.rope_dimension_count = rope_dimension_count
        self.max_seqlen = max_seqlen
        self.use_hf = use_hf
        self.use_table = use_table
        self.dtype = dtype
        self.rope_freq_base = rope_freq_base if rope_freq_base is not None else 10000.0
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_factor = yarn_factor
        self.yarn_original_context_len = yarn_original_context_len
        self.model_arch = model_arch

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        *,
        xt: torch.Tensor,
        start_index: int,
        rotary_embed_table: Optional[torch.Tensor],
    ):
        # freqs_cis shape: max_sl, dim
        # xq_, xk_ shape: bs, sl, _, dim
        xt_ = xt
        _, sl, _, _ = xt_.shape

        if self.model_arch == "llama4":
            freqs_cis_real = rotary_embed_table[0][
                :, : rotary_embed_table[0].shape[1] // 2
            ]
            freqs_cis_imag = rotary_embed_table[1][
                :, : rotary_embed_table[0].shape[1] // 2
            ]
            # TODO: don't use complex numbers as the compiler does better without them.
            freqs_cis = torch.view_as_complex(
                torch.stack([freqs_cis_real, freqs_cis_imag], dim=-1)
            )
            freqs_cis = freqs_cis.unsqueeze(0)
            xt_ = torch.view_as_complex(xt.float().reshape(*xt.shape[:-1], -1, 2))
            xt_out = torch.view_as_real(xt_ * freqs_cis[:, :, None, :]).flatten(3)
            return xt_out.type_as(xt)

        if self.use_hf:
            freqs_cis = rotary_embed_table
            # Slice from max to current sequence length
            cos, sin = [x[start_index : start_index + sl, :] for x in freqs_cis]
            # expand to 1, sl, 1, dim and repeat per bs
            cos = cos[None, :, None, :].repeat(xt.shape[0], 1, 1, 1)
            sin = sin[None, :, None, :].repeat(xt.shape[0], 1, 1, 1)
            xt = xt.transpose(1, 2)
            xt_out = (xt_ * cos) + (self.rotate_half(xt_) * sin)
            return xt_out

        # Offset the table based on starting position.
        if self.use_table:
            freqs_cis = rotary_embed_table[start_index : start_index + sl, :]
            freqs_cis = freqs_cis[0:sl, :]
        else:
            freqs_cis = torch.arange(sl, device=xt.device) + start_index
            freqs_cis = self.compute_rotary_embed_table(freqs_cis)

        assert (
            freqs_cis.shape[0] >= sl
        ), f"Sequence length longer than embedding table ({sl} vs {freqs_cis.shape[0]})"

        freqs_cis = ops.repeat(freqs_cis[None, :, :], (xt_.shape[0], 1, 1))
        xt_out = kernels.apply_rotary_embedding(xt_.to(freqs_cis.dtype), freqs_cis)

        return ops.to(xt_out, xt.dtype)

    def compute_batch_mask(
        self,
        start_positions: Union[torch.Tensor, ReplicatedTensor],
        batch_seq_len: int,
        rotary_embed_table: torch.Tensor,
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
            assert self.use_table, "use_hf requires use_table"
            freqs_cis = rotary_embed_table
            cos, sin = [x[positions_seq.flatten(), :] for x in freqs_cis]
            freqs_cis = (cos[:, None, None, :], sin[:, None, None, :])
            return freqs_cis

        if self.use_table:
            freqs_cis = rotary_embed_table[positions_seq.flatten()]
        else:
            freqs_cis = self.compute_rotary_embed_table(positions_seq.flatten())

        return freqs_cis.unsqueeze(1)

    def apply_batched_mask(self, *, xt: torch.Tensor, mask: torch.Tensor):
        """Applies the embedding to a ragged batch of queries and keys.

        This does a more complicated indexing operation for cases when the each
        sequence in the batch has a potentially different start position.

        positions should be of [bs, sl] and enumerate positions of all tokens.
        """
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim

        if self.use_hf:
            cos, sin = mask
            xt = xt.transpose(1, 2)
            xt_out = (xt * cos) + (self.rotate_half(xt) * sin)
            return xt_out.transpose(1, 2)

        xt_out = kernels.apply_rotary_embedding(xt.to(mask.dtype), mask)

        return xt_out.type_as(xt)

    def _apply_yarn(self, freqs):
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
            inv_freq_llama = torch.where(
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
            freqs = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return freqs

    def compute_rotary_embed_table(self, t):
        dim = self.rope_dimension_count
        if self.use_hf:

            freqs = 1.0 / (
                self.rope_freq_base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
            )
            freqs = self._apply_yarn(freqs)
            freqs = torch.cat((freqs, freqs), dim=-1).to(device=self.device)
            emb = t.unsqueeze(1).float() * freqs.unsqueeze(0).float()

            cos = torch.cos(emb).to(self.dtype)
            sin = torch.sin(emb).to(self.dtype)
            return (cos, sin)

        freqs = 1.0 / (
            self.rope_freq_base ** ((torch.arange(0, dim) // 2).float() / dim * 2.0)
        ).to(device=self.device)
        freqs = self._apply_yarn(freqs)
        freqs = (t.unsqueeze(1) * freqs.unsqueeze(0)).float()
        return freqs

    def create_rotary_embed_table(self):
        t = torch.arange(self.max_seqlen, device=self.device)
        return self.compute_rotary_embed_table(t)
