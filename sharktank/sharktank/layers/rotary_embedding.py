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


def build_rotary_layer(
    rope_dimension_count: int,
    rope_freq_base: Optional[float] = None,
    use_hf: bool = False,
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
    return CachedRotaryLayer(
        rotary_layer=rotary_layer,
        devices=devices,
        dtype=dtype,
        device=device,
    )


# We wrap a shardless rotary layer so the sharding behavior can be handled independently of the numerics.
class CachedRotaryLayer(BaseLayer):
    def __init__(
        self,
        *,
        rotary_layer,
        devices,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._dtype = dtype
        self._rotary_layer = rotary_layer
        self._device = device

    def rotary_embed_table(
        self,
        t: torch.Tensor,
    ) -> tuple[InferenceTensor, InferenceTensor] | InferenceTensor:
        t_0, t_1 = self._rotary_layer.compute_sincos_cache(t, dtype=self._dtype)

        return t_0, t_1

    def forward(
        self,
        *,
        xt: torch.Tensor,
        start_index: int,
    ):
        t = torch.arange(xt.shape[1], device=self._device).unsqueeze(0) + start_index
        table_0, table_1 = self.rotary_embed_table(t)

        return self._rotary_layer(q=xt, sincos_cache=(table_0, table_1))

    def compute_batch_mask(
        self, start_positions: torch.Tensor, batch_seq_len: int
    ) -> tuple[InferenceTensor, InferenceTensor] | InferenceTensor:

        positions_seq = torch.arange(0, batch_seq_len, device=self._device)
        positions_seq = positions_seq.unsqueeze(0) + start_positions.unsqueeze(1)
        table_0, table_1 = self.rotary_embed_table(positions_seq)
        return table_0, table_1

    def apply_batched_mask(
        self, *, xt: torch.Tensor, mask: tuple[InferenceTensor, InferenceTensor]
    ) -> InferenceTensor:
        return self._rotary_layer(q=xt, sincos_cache=mask)
