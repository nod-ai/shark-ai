# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from typing import Optional, Union

import torch

from .base import BaseLayer
from .. import ops
from .. import kernels
from ..types import SplitPrimitiveTensor, ReplicatedTensor, unbox_tensor


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
        tensor_parallelism_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.rope_dimension_count = rope_dimension_count
        self.max_seqlen = max_seqlen
        self.use_hf = use_hf
        self.use_table = use_table
        self.dtype = dtype
        self.rope_freq_base = rope_freq_base if rope_freq_base is not None else 10000.0
        self.tensor_parallelism_size = tensor_parallelism_size

    @property
    def rotary_embed_table(self):
        return self._create_rotary_embed_table()

    def forward(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor],
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
                ]
            )

        if not isinstance(xt, SplitPrimitiveTensor):
            return self.forward_unsharded(
                xt=xt,
                start_index=start_index,
                rotary_embed_table=table,
            )

        assert (
            isinstance(table, ReplicatedTensor) and xt.shard_count == table.shard_count
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
        xt = SplitPrimitiveTensor(ts=xt_shards, shard_dim=xt.shard_dim)
        return xt

    def _create_interleaved_tensor(_, dim):
        """Creates a tensor which indexes an tensor such that
        it alternates between elements of its first and second
        half. Intended for use for HuggingFace's rotation
        implementation.

        Args:
          dim: Size of tensor

        Returns:
          Interleaved indexing tensor
        """
        first_half = torch.arange(dim // 2)
        second_half = torch.arange(dim // 2, dim)

        interleaved_tensor = torch.empty(dim, dtype=torch.long)
        interleaved_tensor[0::2] = first_half
        interleaved_tensor[1::2] = second_half

        return interleaved_tensor

    def _create_ordering_tensor(_, dim):
        """Creates a tensor which indexes an tensor such that
        it reverses the alternation induced by create_interleaved_tesnor.
        Intended for use for HuggingFace's rotation implementation.

        Args:
          dim: Size of tensor

        Returns:
          Ordering indexing tensor
        """
        order_tensor = torch.empty(dim, dtype=torch.long)
        order_tensor[: dim // 2] = torch.arange(0, dim, 2)
        order_tensor[dim // 2 :] = torch.arange(1, dim, 2)
        return order_tensor

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward_unsharded(
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
            freqs_cis = self._compute_rotary_embed_table(freqs_cis)

        assert (
            freqs_cis.shape[0] >= sl
        ), f"Sequence length longer than embedding table ({sl} vs {freqs_cis.shape[0]})"

        freqs_cis = ops.repeat(freqs_cis[None, :, :], (xt_.shape[0], 1, 1))
        xt_out = kernels.apply_rotary_embedding(xt_.to(freqs_cis.dtype), freqs_cis)

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
            assert self.use_table, "use_hf requires use_table"
            freqs_cis = self.rotary_embed_table
            cos, sin = [x[positions_seq.flatten(), :] for x in freqs_cis]
            freqs_cis = (cos[:, None, None, :], sin[:, None, None, :])
            return freqs_cis

        if self.use_table:
            freqs_cis = self.rotary_embed_table[positions_seq.flatten()]
        else:
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
        xt: Union[torch.Tensor, SplitPrimitiveTensor],
        mask: Union[torch.Tensor, ReplicatedTensor],
    ):
        if not isinstance(xt, SplitPrimitiveTensor):
            return self.apply_batched_mask_unsharded(xt=xt, mask=mask)

        assert isinstance(mask, ReplicatedTensor) and mask.shard_count == xt.shard_count
        xt_shards = [
            self.apply_batched_mask_unsharded(
                xt=unbox_tensor(xt_shard),
                mask=unbox_tensor(mask_shard),
            )
            for xt_shard, mask_shard in zip(xt.shards, mask.shards)
        ]
        xt = SplitPrimitiveTensor(ts=xt_shards, shard_dim=xt.shard_dim)
        return xt

    def apply_batched_mask_unsharded(self, *, xt: torch.Tensor, mask: torch.Tensor):
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

    def _compute_rotary_embed_table(self, t):
        dim = self.rope_dimension_count
        if self.use_hf:

            freqs = 1.0 / (
                self.rope_freq_base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
            )
            ### from llama3 embedding changes
            # TODO: get these values from Dataset
            factor = 8  # in the original implementation
            low_freq_factor = 1  # in the original implementation
            high_freq_factor = 4
            old_context_len = 8192

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            inv_freq = freqs
            wavelen = 2 * torch.pi / inv_freq
            inv_freq_llama = torch.where(
                wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
            )

            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

            freqs = torch.cat((freqs, freqs), dim=-1).to(device=self.device)
            emb = t.unsqueeze(1).float() * freqs.unsqueeze(0).float()

            cos = torch.cos(emb).to(self.dtype)
            sin = torch.sin(emb).to(self.dtype)
            return (cos, sin)

        freqs = 1.0 / (
            self.rope_freq_base ** ((torch.arange(0, dim) // 2).float() / dim * 2.0)
        ).to(device=self.device)
        freqs = (t.unsqueeze(1) * freqs.unsqueeze(0)).float()
        return freqs

    def _create_rotary_embed_table(self):
        t = torch.arange(self.max_seqlen, device=self.device)
        freqs_cis = self._compute_rotary_embed_table(t)
        return self._replicate(freqs_cis)

    def _replicate(self, t):
        if self.tensor_parallelism_size > 1:
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t = ops.replicate(t, self.tensor_parallelism_size)

        return t
