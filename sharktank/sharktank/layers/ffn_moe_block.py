# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F

from .base import ThetaLayer
from .linear import LinearLayer
from sharktank.types import Theta, DefaultPrimitiveTensor
from sharktank.ops import einsum_2args, elementwise

__all__ = [
    "FFNMOE",
    "PreGatherFFNMOE",
]


class PreGatherFFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        activation=F.silu,
    ):

        super().__init__(theta)

        # (num_experts, expert_feature_dim, feature_dim)
        self.ffn_gate = theta.tensor("ffn_gate_exps", "weight")

        # (num_experts, expert_feature_dim, feature_dim)
        self.ffn_up = theta.tensor("ffn_up_exps", "weight")

        # (num_experts, feature_dim, expert_feature_dim)
        self.ffn_down = theta.tensor("ffn_down_exps", "weight")

        self.activation = activation

    def pre_matmul_gather_orig(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        # (batch_size * sequence_length, num_top_experts)
        experts: torch.Tensor,
        einstring="mk,menk->men",
    ):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        matmul = einsum_2args(inputs, weights, einstring)
        return matmul

    def pre_matmul_gather(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        # (batch_size * sequence_length, num_top_experts)
        experts: torch.Tensor,
        # (bs * sl, num_top_experts)
        expert_gate: torch.Tensor | None = None,
        einstring="mk,menk->men",
    ):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        if expert_gate is not None:
            weights = weights * expert_gate.view(
                expert_gate.shape[0], expert_gate.shape[1], 1, 1
            )
        matmul = einsum_2args(inputs, weights, einstring)
        return matmul

    def forward(
        self,
        h: torch.Tensor,  # (bs * sl, feature_dim)
        experts: torch.Tensor,  # (bs * sl, num_top_experts)
        expert_gate: torch.Tensor,  # (bs * sl, num_top_experts)
    ):
        # bs: batch_size
        # sl: sequence_length

        # (bs * sl, num_top_experts, expert_feature_dim)
        ffn_gate = self.pre_matmul_gather(h, self.ffn_gate, experts, None)
        ffn_gate = einsum_2args(expert_gate, ffn_gate, "me,men->men")
        ffn_gate = elementwise(self.activation, ffn_gate)

        # (bs * sl, num_top_experts, expert_feature_dim)
        ffn_up = self.pre_matmul_gather(h, self.ffn_up, experts, None)
        ffn_up = einsum_2args(expert_gate, ffn_up, "me,men->men")

        # (bs * sl, num_top_experts, feature_dim)
        ffn_down = self.pre_matmul_gather(
            ffn_gate * ffn_up, self.ffn_down, experts, einstring="mek,menk->men"
        )
        # (bs * sl, num_top_experts, feature_dim)
        # ffn_down = einsum_2args(expert_gate, ffn_down, "me,men->men")
        return torch.sum(ffn_down, dim=1)  # (bs * sl, feature_dim)


class FFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        expert_idx: Optional[int] = None,
    ):

        super().__init__(theta)

        if theta.optional_tensor("ffn_gate_exps") is not None:
            merged_tensor = theta.tensor("ffn_gate_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor,
                layer_name="ffn_gate",
                expert_idx=expert_idx,
            )

            self.add_module("ffn_gate", LinearLayer(Theta({"weight": expert_tensor})))

            merged_tensor = theta.tensor("ffn_up_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor, layer_name="ffn_up", expert_idx=expert_idx
            )

            self.add_module("ffn_up", LinearLayer(Theta({"weight": expert_tensor})))

            merged_tensor = theta.tensor("ffn_down_exps", "weight")

            expert_tensor = extract_ffn_layer(
                merged_tensor=merged_tensor,
                layer_name="ffn_down",
                expert_idx=expert_idx,
            )

            self.add_module("ffn_down", LinearLayer(Theta({"weight": expert_tensor})))

        else:
            self.add_module("ffn_gate", LinearLayer(theta("ffn_gate", expert_idx)))
            self.add_module("ffn_up", LinearLayer(theta("ffn_up", expert_idx)))
            self.add_module("ffn_down", LinearLayer(theta("ffn_down", expert_idx)))

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_gate = F.silu(self.ffn_gate(h))
        ffn_up = self.ffn_up(h)
        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return ffn_down


def extract_ffn_layer(
    merged_tensor: DefaultPrimitiveTensor, layer_name: str, expert_idx: int
):
    # fetches the block_idx from merged_tensor_name. e.g. blk.0.ffn_gate_exps.weight
    expert_layer_name = (
        f"blk.{merged_tensor.name.split('.')[1]}.{layer_name}.{expert_idx}.weight"
    )
    expert_tensor = DefaultPrimitiveTensor(
        name=expert_layer_name, data=merged_tensor.as_torch()[expert_idx]
    )
    return expert_tensor
