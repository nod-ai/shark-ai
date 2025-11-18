# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from amdsharktank import ops
from amdsharktank.types.tensors import AnyTensor, InferenceTensor, ReplicatedTensor

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
        t: AnyTensor,
    ) -> tuple[AnyTensor, AnyTensor]:
        t_0, t_1 = self._rotary_layer.compute_sincos_cache(t, dtype=self._dtype)
        return t_0, t_1

    def forward(
        self,
        *,
        xt: AnyTensor,
        start_positions: Optional[AnyTensor] = None,
    ) -> InferenceTensor:
        batch_seq_len = xt.shape[1]
        mask = self.compute_batch_mask(
            start_positions=start_positions, batch_seq_len=batch_seq_len
        )
        return self.apply_batched_mask(xt=xt, mask=mask)

    def compute_batch_mask(
        self,
        start_positions: Optional[AnyTensor],
        batch_seq_len: int | torch.SymInt,
    ) -> tuple[InferenceTensor, InferenceTensor]:

        positions_seq = ops.arange(0, batch_seq_len, device=self._device)
        positions_seq = positions_seq.unsqueeze(0)
        if start_positions is not None:
            positions_seq = positions_seq + start_positions.unsqueeze(1)
        table_0, table_1 = self._rotary_embed_table(positions_seq)
        return table_0, table_1

    def apply_batched_mask(
        self,
        *,
        xt: AnyTensor,
        mask: tuple[InferenceTensor, InferenceTensor],
    ) -> InferenceTensor:
        return self._rotary_layer(q=xt, sincos_cache=mask)


def build_rotary_layer(
    rope_dimension_count: int,
    rope_freq_base: Optional[float] = None,
    interleave: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    pipeline_stage_to_device_map: list[list[int]] | None = None,
    use_base_frequency_scaling: bool = False,
    **rotary_embd_layer_kwargs,
) -> CachedRotaryLayer:
    rope_freq_base = 10000.0 if rope_freq_base is None else rope_freq_base

    rotary_embd_layer_kwargs = rotary_embd_layer_kwargs.copy()
    rotary_embd_layer_kwargs["rope_theta"] = rope_freq_base
    rotary_embd_layer_kwargs["head_dim"] = rope_dimension_count
    rotary_embd_layer_kwargs["interleaved"] = interleave
    rotary_embd_layer_kwargs["use_base_frequency_scaling"] = use_base_frequency_scaling

    return CachedRotaryLayer(
        rotary_layer=RotaryEmbeddingLayer(**rotary_embd_layer_kwargs),
        dtype=dtype,
        device=device,
    )
