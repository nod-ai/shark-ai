# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Optional

import torch

from sharktank.types.tensors import AnyTensor, InferenceTensor, ReplicatedTensor

from .base import BaseLayer
from .rotary_embedding_hf import RotaryEmbeddingLayer


class CachedRotaryLayer(ABC, BaseLayer):
    @abstractmethod
    def forward(
        self,
        *,
        xt: AnyTensor,
        start_positions: AnyTensor | None = None,
    ) -> AnyTensor:
        ...


class DefaultCachedRotaryLayer(CachedRotaryLayer):
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
        super().__init__()
        self.cached_rotary_layer = DefaultCachedRotaryLayer(
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
        assert (
            len(xt.shards) == 1
        ), "ReplicatedRotaryLayer does not support tensor parallelism"
        if start_positions is not None:
            assert (
                len(start_positions.shards) == 1
            ), "ReplicatedRotaryLayer does not support tensor parallelism"

        devices = xt.devices
        xt = xt.shards[0]
        if start_positions is not None:
            start_positions = start_positions.shards[0]

        rot_embedding = self.cached_rotary_layer.forward(
            xt=xt, start_positions=start_positions
        )
        return ReplicatedTensor(ts=[rot_embedding], devices=devices)


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

    RotaryLayerClazz = DefaultCachedRotaryLayer
    if pipeline_stage_to_device_map and len(pipeline_stage_to_device_map) > 1:
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
