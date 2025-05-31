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
        use_table: bool = True,
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
        self.use_table = use_table
        self.dtype = dtype
        self.rope_freq_base = rope_freq_base if rope_freq_base is not None else 10000.0
        self.tensor_parallelism_size = tensor_parallelism_size
        self.pipeline_parallelism = pipeline_parallelism
        self.devices = (
            devices
            if devices is not None
            else tuple(range(self.tensor_parallelism_size))
        )

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
        # xq_, xk_ shape: bs, sl, _, dim
        xt_ = xt
        _, sl, _, _ = xt_.shape

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

        if self.use_hf:
            xt_out = kernels.apply_rotary_embedding(
                xt_.to(freqs_cis.dtype), freqs_cis, mode="concat"
            )
        else:
            xt_out = kernels.apply_rotary_embedding(
                xt_.to(freqs_cis.dtype), freqs_cis, mode="interleave"
            )

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

        if self.use_hf:
            xt_out = kernels.apply_rotary_embedding(
                xt.to(mask.dtype), mask, mode="concat"
            )
        else:
            xt_out = kernels.apply_rotary_embedding(
                xt.to(mask.dtype), mask, mode="interleave"
            )

        return xt_out.type_as(xt)

    def _compute_rotary_embed_table(self, t):
        dim = self.rope_dimension_count
        if self.use_hf:
            idx = (torch.arange(0, dim) % (dim // 2)).float()
        else:
            idx = (torch.arange(0, dim) // 2).float()

        freqs = 1.0 / self.rope_freq_base ** (idx / dim * 2.0).to(device=self.device)
        freqs = (t.unsqueeze(1) * freqs.unsqueeze(0)).float()
        return freqs

    def _create_rotary_embed_table(self):
        t = torch.arange(self.max_seqlen, device=self.device)
        freqs_cis = self._compute_rotary_embed_table(t)
        return self._replicate(freqs_cis)

    def _replicate(self, t):
        if self.tensor_parallelism_size > 1 or self.pipeline_parallelism:
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t = ops.replicate(t, self.tensor_parallelism_size, devices=self.devices)

        return t
