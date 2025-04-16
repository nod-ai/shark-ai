# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer
from .norm import RMSNormLayer
from .ffn_block import FFN
from .ffn_moe_block import FFNMOE, PreGatherFFNMOE

from ..ops import softmax, topk

__all__ = [
    "MoeBlock",
]


class MoeBlock(ThetaLayer):
    """
    This implementation considers MoE operations as block-sparse
    operations to support imbalanced token assignments to experts.
    This enables the MoE to operate at a faster rate and in full capacity without any dropped tokens
    (or reduced performance).
    """

    def __init__(
        self,
        theta: Theta,
        expert_used_count: int,
        rms_epsilon: float,
        moe_activation=F.silu,
        *,
        score_experts=softmax,
        normalize_experts=True,
        add_residual=True,
        expert_count: Optional[int] = None,
        n_expert_groups: Optional[int] = None,
        n_limited_groups: Optional[int] = None,
        route_scale: Optional[float] = None,
    ):
        super().__init__(theta)
        self.expert_used_count = expert_used_count
        self.expert_count = expert_count
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_experts = score_experts
        self.normalize_experts = normalize_experts
        self.add_residual = add_residual
        self.route_scale = route_scale
        self.ffn_gate_inp = torch.nn.Identity()
        self.ffn_norm = torch.nn.Identity()
        self.layer_output_norm = torch.nn.Identity()
        self.shared_experts = None

        # Add router gate
        if theta.optional_tensor("ffn_gate_inp") is not None:
            self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        # Add FFN norm
        if theta.optional_tensor("ffn_norm") is not None:
            self.ffn_norm = RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)

        if theta.optional_tensor("ffn_gate_shexp") is not None:
            self.shared_experts = FFN(
                theta=theta, activation_fn=moe_activation, rms_epsilon=rms_epsilon
            )

        # Add optional FFN output norm layer
        if theta.optional_tensor("layer_output_norm") is not None:
            self.layer_output_norm = RMSNormLayer(
                theta("layer_output_norm"), epsilon=rms_epsilon
            )

        self.routed_experts = PreGatherFFNMOE(theta, activation=moe_activation)

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # For each token, the router calculates the router weights for all experts
        # router_logits: (batch_size * sequence_length, expert_count)
        router_logits = self.ffn_gate_inp(ffn_input)
        router_weights = self.score_experts(router_logits.to(torch.float))

        # self.n_expert_groups = None
        # self.n_limited_groups = None
        # Select top k experts from router weights
        if self.n_expert_groups is not None and self.n_limited_groups is not None:
            # print('moe here 1')

            scores_for_choice = router_weights.view(-1, self.expert_count)

            group_scores = (
                router_weights.view(
                    -1, self.n_expert_groups, self.expert_count // self.n_expert_groups
                )
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = topk(group_scores, k=self.n_limited_groups, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    -1, self.n_expert_groups, self.expert_count // self.n_expert_groups
                )
                .reshape(-1, self.expert_count)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
            expert_gate, top_k_experts = topk(
                scores_for_choice, k=self.expert_used_count, dim=-1
            )
        else:
            # print('moe here 2')
            expert_gate, top_k_experts = topk(
                router_weights, self.expert_used_count, dim=-1
            )

        if self.normalize_experts:
            expert_gate /= expert_gate.sum(dim=-1, keepdim=True)

        expert_gate = expert_gate.to(ffn_input.dtype)

        if self.route_scale is not None:
            expert_gate = expert_gate * self.route_scale

        moe_output = self.routed_experts(ffn_input, top_k_experts, expert_gate)

        if self.shared_experts:
            moe_output = moe_output + self.shared_experts(ffn_input)

        moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)

        moe_output = self.layer_output_norm(moe_output)
        if self.add_residual:
            moe_output = h + moe_output

        return moe_output
