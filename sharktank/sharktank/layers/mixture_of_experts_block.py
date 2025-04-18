# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types import Theta
from sharktank.layers import *

from sharktank.ops import softmax, topk

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
        moe_activation=torch.nn.functional.silu,
        *,
        score_experts=softmax,
        normalize_experts=True,
        add_residual=True,
        route_scale: Optional[float] = 1.0,
    ):
        super().__init__(theta)
        self.expert_used_count = expert_used_count
        self.score_experts = score_experts
        self.normalize_experts = normalize_experts
        self.add_residual = add_residual

        # Add router gate
        self.add_module("ffn_gate_inp", LinearLayer(theta("ffn_gate_inp")))

        self.ffn_norm = torch.nn.Identity()
        self.layer_output_norm = torch.nn.Identity()
        self.shared_experts = None
        self.route_scale = route_scale if route_scale > 1 else None

        # Add FFN norm
        if "ffn_norm" in theta:
            self.ffn_norm = RMSNormLayer(theta("ffn_norm"), epsilon=rms_epsilon)

        if "shared_experts" in theta:
            self.shared_experts = FFN(
                theta("shared_experts"),
                rms_epsilon=rms_epsilon,
                activation_fn=moe_activation,
                add_residual=False,
            )

        # Add optional FFN output norm layer
        if theta.optional_tensor("layer_output_norm") is not None:
            self.layer_output_norm = RMSNormLayer(
                theta("layer_output_norm"), epsilon=rms_epsilon
            )

        # Add expert_count x FFN
        self.experts = PreGatherFFNMOE(theta, activation=moe_activation)

        from sharktank.types import unbox_tensor

        self.hf_moe = Llama4TextMoe(
            num_experts_per_tok=expert_used_count,
            hidden_size=theta.tensor("ffn_gate_exps", "weight").shape[2],
            num_local_experts=theta.tensor("ffn_gate_exps", "weight").shape[0],
            intermediate_size=theta.tensor("ffn_gate_exps", "weight").shape[1],
            act_fn=moe_activation,
            shared_expert=self.shared_experts,
        )
        self.hf_moe.router.weight.data = unbox_tensor(theta("ffn_gate_inp.weight"))
        self.hf_moe.experts.gate_up_proj.data = torch.cat(
            [
                unbox_tensor(theta("ffn_gate_exps.weight")),
                unbox_tensor(theta("ffn_up_exps.weight")),
            ],
            dim=1,
        ).transpose(1, 2)
        self.hf_moe.experts.down_proj.data = unbox_tensor(
            theta("ffn_down_exps.weight")
        ).transpose(1, 2)

    def forward(
        self,
        # shape: (batch_size, sequence_length, feature_dim)
        h: torch.Tensor,
    ):
        ffn_input = self.ffn_norm(h)

        hf_res = self.hf_moe(ffn_input)[0]
        hf_moe_output = hf_res.reshape(h.shape)

        ###############################################################################
        batch_size, sequence_length, feature_dim = ffn_input.shape
        ffn_input = ffn_input.view(-1, feature_dim)

        # For each token, the router calculates the router weights for all experts
        # shape: (batch_size * sequence_length, expert_count)
        router_logits = self.ffn_gate_inp(ffn_input)
        router_weights = self.score_experts(router_logits.to(torch.float))

        # Select top k experts from router weights
        # shape: (batch_size * sequence_length, expert_used_count)
        expert_gate, top_k_experts = topk(
            router_weights, self.expert_used_count, dim=-1
        )

        if self.normalize_experts:
            expert_gate /= expert_gate.sum(dim=-1, keepdim=True)

        expert_gate = expert_gate.to(ffn_input.dtype)

        if self.route_scale is not None:
            expert_gate = expert_gate * self.route_scale

        # shape: (batch_size * sequence_length, feature_dim)
        moe_output = self.experts(ffn_input, top_k_experts, expert_gate)

        if self.shared_experts:
            moe_output = moe_output + self.shared_experts(ffn_input)

        moe_output = moe_output.reshape(batch_size, sequence_length, feature_dim)
        ###############################################################################

        torch.testing.assert_close(moe_output, hf_moe_output, atol=1e-3, rtol=1e-2)
        moe_output = hf_moe_output

        moe_output = self.layer_output_norm(moe_output)

        if self.add_residual:
            moe_output = h + moe_output

        return moe_output


class Llama4TextMoe(torch.nn.Module):
    def __init__(
        self,
        num_experts_per_tok: int,
        hidden_size: int,
        num_local_experts: int,
        intermediate_size: int,
        act_fn,
        shared_expert: FFN,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.hidden_dim = hidden_size
        self.num_experts = num_local_experts
        self.experts = Llama4TextExperts(
            num_local_experts=num_local_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            act_fn=act_fn,
        )
        self.router = torch.nn.Linear(hidden_size, num_local_experts, bias=False)
        self.shared_expert = shared_expert

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        # (tokens_per_expert, hidden_dim)
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # (num_experts, tokens_per_expert)
        router_logits = self.router(hidden_states).transpose(0, 1)
        tokens_per_expert = batch * seq_len

        # (tokens_per_expert, top_k)
        router_top_value, router_indices = torch.topk(
            router_logits.transpose(0, 1), self.top_k, dim=1
        )
        # (num_experts, tokens_per_expert)
        router_scores = (
            torch.full_like(router_logits.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device)
            .view(1, -1)
            .expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        # (num_experts * tokens_per_expert, hidden_dim)
        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        # (num_experts * tokens_per_expert, hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        # we gather inputs corresponding to each expert based on the router indices
        # (num_experts * tokens_per_expert, hidden_dim)
        routed_in = routed_in * router_scores.reshape(-1, 1)
        # (num_experts * tokens_per_expert, hidden_dim)
        routed_out = self.experts(routed_in)
        # (tokens_per_expert, hidden_dim)
        out = self.shared_expert(hidden_states)
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out.scatter_add_(
            dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim)
        )
        return out, router_scores


class Llama4TextExperts(torch.nn.Module):
    def __init__(
        self, num_local_experts: int, intermediate_size: int, hidden_size: int, act_fn
    ):
        super().__init__()
        self.num_experts = num_local_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
        )
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        # (num_experts, tokens_per_expert, hidden_size)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        # (num_experts, tokens_per_expert, expert_dim * 2)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        # (num_experts, tokens_per_expert, expert_dim)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        # (num_experts, tokens_per_expert, hidden_size)
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states
