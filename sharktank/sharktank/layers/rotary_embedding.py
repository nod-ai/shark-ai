# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import torch

from sharktank.types.tensors import InferenceTensor

from .base import BaseLayer
from .rotary_embedding_hf import RotaryEmbeddingLayer
from sharktank import ops, kernels
from sharktank.types import (
    ShardedTensor,
    SplitPrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    unbox_tensor,
)


def build_rotary_layer(
    rope_dimension_count: int,
    rope_freq_base: Optional[float] = None,
    use_hf: bool = False,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism: bool = False,
    devices=None,
    dtype=torch.float32,
    device: torch.device = None,
    **kwargs,
):
    rope_freq_base = 10000.0 if rope_freq_base is None else rope_freq_base

    rotary_layer = RotaryEmbeddingLayer(
        rope_theta=rope_freq_base,
        head_dim=rope_dimension_count,
        interleaved=not use_hf,
        **kwargs,
    )
    return ShardedRotaryLayer(
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism=pipeline_parallelism,
        rotary_layer=rotary_layer,
        devices=devices,
        dtype=dtype,
        device=device,
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
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._dtype = dtype
        self._pipeline_parallelism = pipeline_parallelism
        self._tensor_parallelism_size = tensor_parallelism_size
        self._rotary_layer = rotary_layer
        self._device = device
        self._devices = (
            devices
            if devices is not None
            else tuple(range(self._tensor_parallelism_size))
        )

    def rotary_embed_table(
        self, t: torch.Tensor, ignore_sharding: bool = False
    ) -> tuple[InferenceTensor, InferenceTensor] | InferenceTensor:
        t_0, t_1 = self._rotary_layer.compute_sincos_cache(t, dtype=self._dtype)
        if not ignore_sharding and (
            self._tensor_parallelism_size > 1 or self._pipeline_parallelism
        ):
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t_0 = ops.replicate(
                t_0, self._tensor_parallelism_size, devices=self._devices
            )
            t_1 = ops.replicate(
                t_1, self._tensor_parallelism_size, devices=self._devices
            )

        return t_0, t_1

    def forward(
        self,
        *,
        xt: Union[torch.Tensor, ShardedTensor],
        start_index: int,
    ):
        t = torch.arange(xt.shape[1], device=self._device).unsqueeze(0) + start_index
        table_0, table_1 = self.rotary_embed_table(t)

        if not isinstance(table_0, ShardedTensor):
            return self._rotary_layer(q=xt, sincos_cache=(table_0, table_1))

        shards = [
            self._rotary_layer(q=xs, sincos_cache=(ts_0, ts_1))
            for xs, ts_0, ts_1 in zip(xt.shards, table_0.shards, table_1.shards)
        ]
        return xt.clone(ts=shards)

    def compute_batch_mask(
        self, start_positions: Union[torch.Tensor, ShardedTensor], batch_seq_len: int
    ) -> tuple[InferenceTensor, InferenceTensor] | InferenceTensor:

        if isinstance(start_positions, ShardedTensor):
            shards_0 = []
            shards_1 = []
            for ss in start_positions.shards:
                positions_seq = torch.arange(0, batch_seq_len, device=self._device)
                positions_seq = positions_seq.unsqueeze(0) + ss.unsqueeze(1)
                table_0, table_1 = self.rotary_embed_table(
                    positions_seq, ignore_sharding=True
                )
                shards_0.append(table_0)
                shards_1.append(table_1)

            return start_positions.clone(ts=shards_0), start_positions.clone(
                ts=shards_1
            )

        positions_seq = torch.arange(0, batch_seq_len, device=self._device)
        positions_seq = positions_seq.unsqueeze(0) + start_positions.unsqueeze(1)
        table_0, table_1 = self.rotary_embed_table(positions_seq)
        return table_0, table_1

    def apply_batched_mask(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor, ReplicatedTensor],
        mask: tuple[InferenceTensor, InferenceTensor] | InferenceTensor,
    ) -> Union[SplitPrimitiveTensor, ReplicatedTensor]:
        if not isinstance(xt, ShardedTensor):
            return self._rotary_layer(q=xt, sincos_cache=mask)

        assert (
            isinstance(mask[0], ReplicatedTensor)
            and mask[0].shard_count == xt.shard_count
        )
        assert (
            isinstance(mask[1], ReplicatedTensor)
            and mask[1].shard_count == xt.shard_count
        )
        xt_shards = [
            self._rotary_layer(
                q=unbox_tensor(xt_shard),
                sincos_cache=(
                    unbox_tensor(mask_sin_shard),
                    unbox_tensor(mask_cos_shard),
                ),
            )
            for xt_shard, mask_sin_shard, mask_cos_shard in zip(
                xt.shards, mask[0].shards, mask[1].shards
            )
        ]
        return xt.clone(ts=xt_shards)
