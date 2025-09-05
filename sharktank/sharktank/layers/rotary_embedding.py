# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types.tensors import InferenceTensor, ReplicatedTensor

from .base import BaseLayer
from .rotary_embedding_hf import RotaryEmbeddingLayer


class CachedRotaryLayer(BaseLayer):
    def __init__(
        self,
        *,
        rotary_layer: RotaryEmbeddingLayer,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self._dtype = dtype
        self._rotary_layer = rotary_layer
        self._device = device

    def _rotary_embed_table(
        self,
        t: torch.Tensor,
    ) -> tuple[InferenceTensor, InferenceTensor]:
        t_0, t_1 = self._rotary_layer.compute_sincos_cache(t, dtype=self._dtype)
        return t_0, t_1

    def forward(
        self,
        *,
        xt: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
    ) -> InferenceTensor:
        batch_seq_len = xt.shape[1]
        mask = self.compute_batch_mask(
            start_positions=start_positions, batch_seq_len=batch_seq_len
        )
        return self.apply_batched_mask(xt=xt, mask=mask)

    def compute_batch_mask(
        self,
        start_positions: Optional[torch.Tensor],
        batch_seq_len: int | torch.SymInt,
        devices: list[int] | None = None,
    ) -> tuple[InferenceTensor, InferenceTensor]:

        positions_seq = torch.arange(0, batch_seq_len, device=self._device)
        positions_seq = positions_seq.unsqueeze(0)
        if start_positions is not None:
            positions_seq = positions_seq + start_positions.unsqueeze(1)
        table_0, table_1 = self._rotary_embed_table(positions_seq)
        return table_0, table_1

    def apply_batched_mask(
        self,
        *,
        xt: torch.Tensor,
        mask: tuple[InferenceTensor, InferenceTensor],
    ) -> InferenceTensor:
        return self._rotary_layer(q=xt, sincos_cache=mask)


class ReplicatedRotaryLayer(CachedRotaryLayer):
    def __init__(
        self,
        *,
        rotary_layer: RotaryEmbeddingLayer,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__(
            rotary_layer=rotary_layer,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        *,
        xt: ReplicatedTensor,
        start_positions: ReplicatedTensor | None = None,
    ) -> InferenceTensor:
        batch_seq_len = xt.shape[1]
        mask = self.compute_batch_mask(
            start_positions=start_positions,
            batch_seq_len=batch_seq_len,
            devices=xt.devices,
        )
        return self.apply_batched_mask(xt=xt, mask=mask)

    def compute_batch_mask(
        self,
        start_positions: ReplicatedTensor | None,
        batch_seq_len: int | torch.SymInt,
        devices: list[int] | None = None,
    ) -> tuple[ReplicatedTensor, ReplicatedTensor]:
        assert (start_positions is not None) or (
            devices is not None
        ), "Either start_positions or devices must be provided to properly place the replicated tensor"

        if devices is not None and start_positions is not None:
            assert list(devices) == list(
                start_positions.devices
            ), "Devices must match between provided devices and start_positions"

        if devices is None:
            devices = start_positions.devices

        if start_positions is None:
            # Create from a single shard and introduce transfers since there's no start positions to hint what device they should go on
            # TODO: Is this needed, or will IREE figure out what device to put it on based downstream uses?
            t0_shard, t1_shard = super().compute_batch_mask(
                start_positions=None, batch_seq_len=batch_seq_len
            )
            table_0 = ReplicatedTensor(
                ts=t0_shard, shard_count=len(devices), devices=devices
            )
            table_1 = ReplicatedTensor(
                ts=t1_shard, shard_count=len(devices), devices=devices
            )
        else:
            table_0_shards, table_1_shards = [], []
            for start_position_shard in start_positions.shards:
                t0_shard, t1_shard = super().compute_batch_mask(
                    start_position_shard, batch_seq_len
                )
                table_0_shards.append(t0_shard)
                table_1_shards.append(t1_shard)
            table_0 = ReplicatedTensor(ts=table_0_shards, devices=devices)
            table_1 = ReplicatedTensor(ts=table_1_shards, devices=devices)
        return table_0, table_1

    def apply_batched_mask(
        self,
        *,
        xt: ReplicatedTensor,
        mask: tuple[ReplicatedTensor, ReplicatedTensor],
    ) -> ReplicatedTensor:
        assert list(xt.devices) == list(mask[0].devices) == list(mask[1].devices)
        assert len(xt.devices) == len(mask[0].devices) == len(mask[1].devices) == 1

        mask = (mask[0].shards[0], mask[1].shards[0])
        shard = self._rotary_layer(q=xt.shards[0], sincos_cache=mask)
        return ReplicatedTensor(ts=[shard], devices=xt.devices)


def build_rotary_layer(
    rope_dimension_count: int,
    rope_freq_base: Optional[float] = None,
    interleave: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    pipeline_stage_to_device_map: list[list[int]] | None = None,
    **rotary_embd_layer_kwargs,
) -> CachedRotaryLayer:
    rope_freq_base = 10000.0 if rope_freq_base is None else rope_freq_base

    rotary_embd_layer_kwargs = rotary_embd_layer_kwargs.copy()
    rotary_embd_layer_kwargs["rope_theta"] = rope_freq_base
    rotary_embd_layer_kwargs["head_dim"] = rope_dimension_count
    rotary_embd_layer_kwargs["interleaved"] = interleave

    RotaryLayerClazz = CachedRotaryLayer
    if pipeline_stage_to_device_map is not None:
        num_shards = len(pipeline_stage_to_device_map[0])
        if num_shards == 1:
            RotaryLayerClazz = ReplicatedRotaryLayer
        else:
            raise NotImplementedError("Tensor parallelism not supported")

    return RotaryLayerClazz(
        rotary_layer=RotaryEmbeddingLayer(**rotary_embd_layer_kwargs),
        dtype=dtype,
        device=device,
    )
