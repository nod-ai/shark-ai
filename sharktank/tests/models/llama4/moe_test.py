from amdsharktank.utils.testing import TempDirTestBase
from amdsharktank.layers.ffn_block import FFN
import torch
from amdsharktank import ops


class Llama4Test(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)

    def test_moe(self):
        from amdsharktank.layers.testing import make_random_moe_block_theta
        from amdsharktank.layers import MoeBlock

        dtype = torch.float32
        feature_dim = 7
        expert_hidden_dim = 3
        num_experts = 5
        expert_used_count = 2
        num_shared_experts = 11
        shared_expert_hidden_dim = 13
        batch_size = 17
        sequence_length = 19

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            with_layer_output_norm=False,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )

        moe_block = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            expert_used_count=expert_used_count,
            rms_epsilon=0.01,
            moe_activation=torch.nn.functional.silu,
            experts_ffn_moe_block="PreGatherFFNMOE",
            score_experts=torch.nn.functional.sigmoid,
            normalize_experts=False,
            expert_shared_count=num_shared_experts,
            model_arch="llama4",
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        moe_output = moe_block(input)

        from amdsharktank.types import unbox_tensor, Theta

        shared_ffn_theta = theta
        if theta.optional_tensor("ffn_gate_shexp") is not None:
            shared_ffn_theta = Theta(
                {
                    "ffn_gate": theta("ffn_gate_shexp").tree,
                    "ffn_up": theta("ffn_up_shexp").tree,
                    "ffn_down": theta("ffn_down_shexp").tree,
                }
            )
        shared_experts = FFN(
            theta=shared_ffn_theta,
            activation_fn=torch.nn.functional.silu,
        )

        hf_moe = Llama4TextMoe(
            num_experts_per_tok=expert_used_count,
            hidden_size=theta.tensor("ffn_gate_exps", "weight").shape[2],
            num_local_experts=theta.tensor("ffn_gate_exps", "weight").shape[0],
            intermediate_size=theta.tensor("ffn_gate_exps", "weight").shape[1],
            act_fn=torch.nn.functional.silu,
            shared_expert=shared_experts,
        )
        hf_moe.router.weight.data = unbox_tensor(theta("ffn_gate_inp.weight"))
        hf_moe.experts.gate_up_proj.data = ops.cat(
            [
                unbox_tensor(theta("ffn_gate_exps.weight")),
                unbox_tensor(theta("ffn_up_exps.weight")),
            ],
            dim=1,
        ).transpose(1, 2)
        hf_moe.experts.down_proj.data = unbox_tensor(
            theta("ffn_down_exps.weight")
        ).transpose(1, 2)

        hf_moe.shared_expert.gate_proj.weight.data = unbox_tensor(
            theta("ffn_gate_shexp.weight")
        )
        hf_moe.shared_expert.up_proj.weight.data = unbox_tensor(
            theta("ffn_up_shexp.weight")
        )
        hf_moe.shared_expert.down_proj.weight.data = unbox_tensor(
            theta("ffn_down_shexp.weight")
        )

        hf_res = hf_moe(input)[0]
        hf_moe_output = hf_res.reshape(input.shape)
        torch.testing.assert_close(moe_output, hf_moe_output, atol=1e-3, rtol=1e-2)


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
            num_local_experts, intermediate_size, hidden_size
        )
        self.router = torch.nn.Linear(hidden_size, num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states).transpose(0, 1)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(
            router_logits.transpose(0, 1), self.top_k, dim=1
        )
        router_scores = (
            torch.full_like(router_logits.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        router_scores = ops.sigmoid(router_scores.to(torch.float32)).to(
            hidden_states.dtype
        )

        routed_in = hidden_states.repeat(self.num_experts, 1)
        routed_out = self.experts(routed_in).view(
            self.num_experts, -1, self.hidden_dim
        )  # (E, T, H)

        # llama4 semantics: p applied inside & outside → p^2
        p_sq = (router_scores * router_scores).view(
            self.num_experts, -1, 1
        )  # (E, T, 1)
        experts_combined = (routed_out * p_sq).sum(0)  # (T, H)

        shared_out = self.shared_expert(hidden_states)  # (T, H)
        out = shared_out + experts_combined
        return out, router_scores


class Llama4TextExperts(torch.nn.Module):
    def __init__(self, num_local_experts, intermediate_size, hidden_size):
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
        self.act_fn = torch.nn.SiLU()

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
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


# Phi3MLP
class Llama4TextMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = intermediate_size

        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(down_proj)
