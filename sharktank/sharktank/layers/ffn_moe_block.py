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
from amdsharktank.types import DefaultPrimitiveTensor, Theta
from amdsharktank.ops import einsum_2args, elementwise
from amdsharktank import ops

__all__ = [
    "DenseFFNMOE",
    "SparseFFNMOE",
    "PreGatherFFNMOE",
]


class PreGatherFFNMOE(ThetaLayer):
    """
    for each selected expert e:
        u_e = W_up[e] * h
        g_e = activation(W_gate[e] * h)
        z_e = W_down[e] * (u_e * g_e)
    Output: Y = Σ_e p_e z_e
    """

    def __init__(
        self,
        theta: Theta,
        activation_fn=F.silu,
        model_arch: Optional[str] = None,
        use_moe_swiglu: bool = False,
    ):

        super().__init__(theta)

        # (num_experts, expert_feature_dim, feature_dim)
        self.ffn_gate = theta.tensor("ffn_gate", "weight")

        # (num_experts, expert_feature_dim, feature_dim)
        self.ffn_up = theta.tensor("ffn_up", "weight")

        # (num_experts, feature_dim, expert_feature_dim)
        self.ffn_down = theta.tensor("ffn_down", "weight")

        self.ffn_gate_bias = theta.optional_tensor("ffn_gate", "bias")
        self.ffn_up_bias = theta.optional_tensor("ffn_up", "bias")
        self.ffn_down_bias = theta.optional_tensor("ffn_down", "bias")

        self.activation_fn = activation_fn

        self.model_arch = model_arch
        self.use_moe_swiglu = use_moe_swiglu

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
        matmul = einsum_2args(inputs, weights, einstring)
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

    def _broadcast_bias(self, bias, experts):
        # we need to broadcase the bias because of the issue op.gather issue with vector indexing iree issue: https://github.com/iree-org/iree/issues/22145
        batch_size, num_experts_per_token = experts.shape
        num_experts = bias.shape[0]
        experts_flat = experts.reshape(-1)  # (batch_size * num_experts_per_token,)
        one_hot_flat = torch.nn.functional.one_hot(
            experts_flat, num_classes=num_experts
        ).to(bias.dtype)
        one_hot = one_hot_flat.reshape(batch_size, num_experts_per_token, num_experts)
        # Matrix multiply to select bias values
        return ops.matmul(one_hot, bias)

    def forward(
        self,
        h: torch.Tensor,  # (bs * sl, feature_dim)
        experts: torch.Tensor,  # (bs * sl, num_top_experts)
        expert_gate: torch.Tensor,  # (bs * sl, num_top_experts)
    ):
        """
        for each token h and its selected experts e
            u_e = W_up[e] @ h_t
            g_e = activation(W_gate[e] * h_t)
            z_e = W_down[e] @ (u_e * g_e)

        Output: y_t = Σ_e p_{t,e} * z_e
        Fused implementation: project once with (W_gate || W_up) to get [a_e || b_e],
        then act(a_e)*b_e, then W_down, then weighted sum.

        Note: model_arch == "llama4" applies an extra pre-down scaling.
        """
        # bs: batch_size
        # sl: sequence_length
        gate_w = self.ffn_gate[experts, :, :]  # (N,K,C,D)
        up_w = self.ffn_up[experts, :, :]  # (N,K,C,D)

        h_expanded = h.unsqueeze(1).unsqueeze(1)
        gate_proj = (gate_w * h_expanded).sum(-1)
        up_proj = (up_w * h_expanded).sum(-1)

        if self.ffn_gate_bias is not None:
            gate_proj = gate_proj + self._broadcast_bias(self.ffn_gate_bias, experts)

        if self.ffn_up_bias is not None:
            up_proj = up_proj + self._broadcast_bias(self.ffn_up_bias, experts)

        if self.use_moe_swiglu:
            # Concatenate after separate projections for swiglu
            proj = ops.cat([gate_proj, up_proj], dim=-1)  # (B, K, 2C)
            hidden = ops.swiglu(proj)
        else:
            gate_act = elementwise(self.activation_fn, gate_proj)
            hidden = gate_act * up_proj

        if self.model_arch == "llama4":
            hidden = hidden * expert_gate.unsqueeze(-1)

        down_w = self.ffn_down[experts, :, :]  # (N,K,D_out,C)
        t2 = (down_w * hidden.unsqueeze(2)).sum(-1)

        if self.ffn_down_bias is not None:
            t2 = t2 + self._broadcast_bias(self.ffn_down_bias, experts)  # (N,K,D_out)
        out = (t2 * expert_gate.unsqueeze(-1)).sum(1)
        return out


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
        expert_count: int,
        is_gated: bool = True,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        fake_quant: bool = False,
    ):
        super().__init__(theta)
        self.num_experts = expert_count
        self.ffn = FFN(
            theta,
            is_gated=is_gated,
            activation_fn=activation_fn,
            fake_quant=fake_quant,
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

        router_scores = ops.reshard_like(
            torch.empty([num_tokens, self.num_experts], device=h.device), like=h
        )
        # (self.num_experts, num_tokens)
        router_scores = (
            ops.zeros_like(router_scores, dtype=h.dtype)
            .scatter_(1, top_experts_index, expert_gate)
            .transpose(0, 1)
        )

        # (self.num_experts, num_tokens)
        router_indices = (
            ops.reshard_like(ops.arange(num_tokens, device=h.device), router_scores)
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
        routed_out = ops.reshard_like(routed_out, like=h)

        # (num_tokens, input_feature_dim)
        return routed_out.view(self.num_experts, num_tokens, input_feature_dim).sum(
            dim=0
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
