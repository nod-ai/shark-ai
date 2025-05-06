# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .base import ThetaLayer
from .linear import LinearLayer
from . import FFN
from sharktank.types import AnyTensor, DefaultPrimitiveTensor, Theta
from sharktank.ops import einsum_2args, elementwise
from sharktank import ops

__all__ = [
    "DenseFFNMOE",
    "SparseFFNMOE",
    "PreGatherFFNMOE",
]


class PreGatherFFNMOE(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        activation=F.silu,
    ):

        super().__init__(theta)

        self.ffn_gate = theta.tensor("ffn_gate_exps", "weight")
        self.ffn_up = theta.tensor("ffn_up_exps", "weight")
        self.ffn_down = theta.tensor("ffn_down_exps", "weight")
        self.activation = activation

    def pre_matmul_gather(self, inputs, weights, experts, einstring="mk,menk->men"):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        matmul = einsum_2args(inputs, weights, einstring)
        return matmul

    def bigger_mmg(self, inputs, weights, experts):
        inputs = inputs[:, :]
        weights = weights[experts, :, :]
        matmul = einsum_2args(inputs, weights, "mek,menk->men")
        return matmul

    def one_hot_matmul(self, inputs, weights, experts):
        matmul = einsum_2args(inputs, weights, "mk,bnk->bmn")
        # Post mix the experts
        oh = (
            torch.nn.functional.one_hot(experts.reshape(-1), num_classes=8)
            .transpose(0, 1)
            .to(torch.float32)
        )
        output = einsum_2args(oh, matmul, "bm,bmn->mn")
        return output

    def forward(
        self,
        h: torch.Tensor,
        experts: torch.Tensor,
        expert_gate: torch.Tensor,
    ):
        ffn_gate = self.pre_matmul_gather(h, self.ffn_gate, experts)
        ffn_gate = elementwise(self.activation, ffn_gate)

        ffn_up = self.pre_matmul_gather(h, self.ffn_up, experts)
        ffn_down = self.pre_matmul_gather(
            ffn_gate * ffn_up, self.ffn_down, experts, einstring="mek,menk->men"
        )
        ffn_down = einsum_2args(expert_gate, ffn_down, "me,men->men")
        return torch.sum(ffn_down, dim=1)


class DenseFFNMOE(ThetaLayer):
    """Do not route the tokens to only the top selected experts, but route to all
    experts.

    This would result in doing useless computation for some token expert pairs where
    the routing score is 0, but we would perform fewer number matrix multiplication.
    What approach is better would depend on the number experts and the number of top
    experts.
    """

    def __init__(
        self,
        theta: Theta,
        rms_epsilon: float | None = None,
        is_gated: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        activation_dtype: Optional[torch.dtype] = None,
        fake_quant: bool = False,
    ):
        super().__init__(theta)
        self.num_experts = theta("ffn_gate_exps", "weight").shape[0]
        ffn_theta = Theta(
            {
                "ffn_gate": theta("ffn_gate_exps").tree,
                "ffn_up": theta("ffn_up_exps").tree,
                "ffn_down": theta("ffn_down_exps").tree,
            }
        )
        self.ffn = FFN(
            ffn_theta,
            rms_epsilon=rms_epsilon,
            is_gated=is_gated,
            activation_fn=activation_fn,
            activation_dtype=activation_dtype,
            fake_quant=fake_quant,
            add_residual=False,
        )

    def forward(
        self,
        h: torch.Tensor,
        top_experts_index: torch.Tensor,
        expert_gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:
            Input tokens.
            Shape of (batch_size * sequence_length, input_feature_dim).
        top_experts_index:
            indexes of selected top experts for each token.
            Shape of (batch_size * sequence_length, num_top_experts).
            Value range is [0, num_of_experts)
        expert_gate:
            router weights of selected top experts for each token.
            Shape of (batch_size * sequence_length, num_top_experts).

        Returns:
            Tensor of shape (batch_size * sequence_length, expert_feature_dim)
        """
        num_tokens, input_feature_dim = h.shape

        # (self.num_experts, num_tokens)
        router_scores = (
            torch.full([num_tokens, self.num_experts], 0, dtype=h.dtype)
            .scatter_(1, top_experts_index, expert_gate)
            .transpose(0, 1)
        )

        # (self.num_experts, num_tokens)
        router_indices = (
            torch.arange(num_tokens, device=h.device)
            .view(1, -1)
            .expand(self.num_experts, -1)
        )
        # (self.num_experts * num_tokens, input_feature_dim)
        router_indices = router_indices.reshape(-1, 1).expand(-1, input_feature_dim)

        # (self.num_experts * num_tokens, input_feature_dim)
        routed_in = ops.gather(
            input=h,
            dim=0,
            index=router_indices,
        )
        routed_in = routed_in * (router_scores > 0).reshape(-1, 1)
        routed_in = routed_in.view(self.num_experts, num_tokens, input_feature_dim)

        # (self.num_experts * num_tokens, input_feature_dim)
        routed_out = self.ffn(routed_in).view(
            self.num_experts * num_tokens, input_feature_dim
        )
        routed_out = routed_out * router_scores.reshape(-1, 1)

        # (num_tokens, input_feature_dim)
        return ops.zeros_like(h).scatter_add(
            dim=0, index=router_indices, src=routed_out
        )


class SparseFFNMOE(ThetaLayer):
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
