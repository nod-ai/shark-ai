# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from parameterized import parameterized, param
from typing import Callable
import pytest
import torch
from iree.turbine.aot import *
from amdsharktank.layers.testing import make_random_moe_block_theta
from amdsharktank.utils.random import make_rand_torch
from amdsharktank.utils.testing import assert_tensor_close
from amdsharktank.layers.mixture_of_experts_block import MoeBlock
from amdsharktank.types.sharding import MoeBlockSharding
from amdsharktank.ops import reshard, reshard_like, replicate, swiglu, cat
from amdsharktank.types import unbox_tensor
from amdsharktank.layers.mixture_of_experts_block import PreGatherFFNMOE


from amdsharktank.types.theta import Theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor


class MoeBlockTest(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123)

    def testExport(self):
        dtype = torch.float32
        batch_size = 3
        seq_len = 5
        in_dim = 7

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=in_dim,
            expert_hidden_dim=13,
            num_experts=17,
            with_ffn_norm=True,
            num_shared_experts=19,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )
        theta.rename_tensors_to_paths()
        model = MoeBlock(
            theta=theta,
            expert_count=17,
            expert_used_count=2,
            rms_epsilon=1e-5,
        )
        fxb = FxProgramsBuilder(model)
        input = make_rand_torch((batch_size, seq_len, in_dim))

        @fxb.export_program(name="moe_block", args=(input,), strict=False)
        def _(model, input: torch.Tensor) -> torch.Tensor:
            return model(input)

    @parameterized.expand(
        [
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=1,
                expert_used_count=1,
                n_expert_groups=None,
                n_limited_groups=None,
                num_shared_experts=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=1.234,
                atol=1e-5,
                rtol=1e-5,
            ),
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=2,
                n_expert_groups=None,
                n_limited_groups=None,
                expert_used_count=1,
                num_shared_experts=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=1.234,
                atol=1e-5,
                rtol=1e-5,
            ),
            param(
                dtype=torch.float32,
                feature_dim=1,
                expert_hidden_dim=1,
                num_experts=3,
                n_expert_groups=None,
                n_limited_groups=None,
                expert_used_count=2,
                num_shared_experts=1,
                batch_size=1,
                sequence_length=1,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=1.234,
                atol=1e-5,
                rtol=1e-5,
            ),
            param(
                dtype=torch.float32,
                feature_dim=2,
                expert_hidden_dim=3,
                num_experts=4,
                n_expert_groups=2,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=2,
                batch_size=2,
                sequence_length=3,
                rms_epsilon=0.03,
                moe_activation_fn=torch.nn.functional.gelu,
                score_experts_fn=torch.nn.functional.softmax,
                normalize_experts=True,
                route_scale=3.21,
                atol=1e-5,
                rtol=1e-5,
            ),
            param(
                dtype=torch.bfloat16,
                feature_dim=7,
                expert_hidden_dim=3,
                num_experts=12,
                n_expert_groups=3,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=11,
                batch_size=17,
                sequence_length=19,
                rms_epsilon=0.01,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=False,
                route_scale=None,
                atol=1e-2,
                rtol=1e-2,
            ),
        ]
    )
    def testParityOfExpertPreGatherFfnAndDenseFfn(
        self,
        dtype: torch.dtype,
        feature_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        n_expert_groups: int | None,
        n_limited_groups: int | None,
        expert_used_count: int,
        num_shared_experts: int,
        batch_size: int,
        sequence_length: int,
        rms_epsilon: float,
        moe_activation_fn: Callable[[torch.Tensor], torch.Tensor],
        score_experts_fn: Callable[[torch.Tensor], torch.Tensor],
        normalize_experts: bool,
        route_scale: float,
        atol: float,
        rtol: float,
    ):
        from amdsharktank.layers.testing import make_random_moe_block_theta
        from amdsharktank.layers import MoeBlock

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            num_shared_experts=num_shared_experts,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )

        moe_with_pre_gather_ffn = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            experts_ffn_moe_block="PreGatherFFNMOE",
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )
        moe_with_dense_ffn = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            experts_ffn_moe_block="DenseFFNMOE",
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        res_pre_gather = moe_with_pre_gather_ffn(input)
        res_dense = moe_with_dense_ffn(input)
        assert_tensor_close(res_pre_gather, res_dense, atol=atol, rtol=rtol)

    @parameterized.expand(
        [
            param(
                dtype=torch.float32,
                feature_dim=7,
                expert_hidden_dim=3,
                num_experts=12,
                n_expert_groups=4,
                n_limited_groups=2,
                expert_used_count=2,
                num_shared_experts=5,
                batch_size=8,
                sequence_length=9,
                rms_epsilon=0.01,
                moe_activation_fn=torch.nn.functional.silu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=None,
                tensor_parallelism_size=2,
            ),
            param(
                dtype=torch.bfloat16,
                feature_dim=2,
                expert_hidden_dim=10,
                num_experts=9,
                n_expert_groups=3,
                n_limited_groups=3,
                expert_used_count=7,
                num_shared_experts=8,
                batch_size=2,
                sequence_length=3,
                rms_epsilon=0.02,
                moe_activation_fn=torch.nn.functional.gelu,
                score_experts_fn=torch.nn.functional.sigmoid,
                normalize_experts=True,
                route_scale=1.1,
                tensor_parallelism_size=3,
            ),
        ]
    )
    def testTensorParallel(
        self,
        dtype: torch.dtype,
        feature_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        n_expert_groups: int | None,
        n_limited_groups: int | None,
        expert_used_count: int,
        num_shared_experts: int,
        batch_size: int,
        sequence_length: int,
        rms_epsilon: float,
        moe_activation_fn: Callable[[torch.Tensor], torch.Tensor],
        score_experts_fn: Callable[[torch.Tensor], torch.Tensor],
        normalize_experts: bool,
        route_scale: float,
        tensor_parallelism_size: int,
    ):
        from amdsharktank.layers.testing import make_random_moe_block_theta
        from amdsharktank.layers import MoeBlock

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=False,
            num_shared_experts=num_shared_experts,
            with_layer_output_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )
        model_arch = "grok"
        if num_shared_experts > 0:
            model_arch = "deepseek2"
        theta_sharding_spec = MoeBlockSharding(
            shard_count=tensor_parallelism_size, model_arch=model_arch
        )
        sharded_theta = reshard(theta, spec=theta_sharding_spec)

        block = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )
        sharded_block = MoeBlock(
            theta=sharded_theta,
            expert_count=num_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_used_count=expert_used_count,
            rms_epsilon=rms_epsilon,
            moe_activation=moe_activation_fn,
            score_experts=score_experts_fn,
            normalize_experts=normalize_experts,
            route_scale=route_scale,
        )

        input = (
            torch.rand([batch_size, sequence_length, feature_dim], dtype=dtype) - 0.5
        )
        sharded_input = replicate(input, count=tensor_parallelism_size)
        expected = block(input)
        actual = sharded_block(sharded_input)
        assert_tensor_close(actual, expected)

    def test_topk_then_softmax_routing(self):
        """
        Regression test for topk_then_softmax routing (GPT-OSS configuration).
        """
        torch.manual_seed(42)

        feature_dim = 8
        expert_hidden_dim = 16
        num_experts = 4
        expert_used_count = 2
        batch_size = 2
        sequence_length = 3
        dtype = torch.float32

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )
        captured = {}

        def score_experts_record(topk_logits, dim=-1):
            # Capture the logits and resulting probs for assertions.
            captured["logits"] = topk_logits.detach()
            probs = torch.softmax(topk_logits, dim=dim)
            captured["probs"] = probs.detach()
            return probs

        moe = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            expert_used_count=expert_used_count,
            rms_epsilon=1e-5,
            topk_then_softmax=True,
            score_experts=score_experts_record,
        )

        input_tensor = torch.randn(
            [batch_size, sequence_length, feature_dim], dtype=dtype
        )

        output = moe(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(torch.isfinite(output).all(), "Output is finite")

        topk_logits = captured["logits"]
        probs = captured["probs"]

        self.assertEqual(
            topk_logits.shape[-1], expert_used_count, "Expected top-k logits tensor"
        )
        self.assertTrue(
            torch.all(topk_logits[..., :-1] >= topk_logits[..., 1:]),
            "Top-k logits are not sorted descending",
        )

        diffs = topk_logits[..., :-1] - topk_logits[..., 1:]
        self.assertTrue(
            torch.all(diffs > 0),
            "Top-k logits not strictly descending (ties or reordering introduced)",
        )
        self.assertTrue(
            torch.allclose(
                probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-6
            ),
            "Gating probabilities do not sum to 1",
        )

    @parameterized.expand(
        [
            param(dtype=torch.float32),
            param(dtype=torch.bfloat16),
            param(dtype=torch.float16),
        ]
    )
    def test_moe_block_dtype_preservation(self, dtype):
        """Test that MoeBlock preserves input dtype through the forward pass."""
        torch.manual_seed(42)

        feature_dim = 8
        expert_hidden_dim = 16
        num_experts = 4
        expert_used_count = 2
        batch_size = 2
        sequence_length = 3

        theta = make_random_moe_block_theta(
            block_idx=0,
            in_dim=feature_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            with_ffn_norm=True,
            dtype_rest=dtype,
            dtype_norm=dtype,
        )

        moe = MoeBlock(
            theta=theta,
            expert_count=num_experts,
            expert_used_count=expert_used_count,
            rms_epsilon=1e-5,
        )

        input_tensor = torch.randn(
            batch_size, sequence_length, feature_dim, dtype=dtype
        )

        output = moe(input_tensor)

        # Verify dtype is preserved
        self.assertEqual(output.dtype, dtype, f"Expected {dtype}, got {output.dtype}")
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(
            torch.isfinite(output).all(),
            f"Output does not contain infinite for dtype {dtype}",
        )


# =================== MoE Golden Test =====================
"""
Analytic MOE equation used in these golden tests
Each expert e return : Expert_e(x)= scale_e * x (scale_e constant across tokens)
Router produces probabilities p_e over selected experts
Combined output for a token is
    y = Σ_e(p_e * Expert_e(x)) = Σ_e (p_e * scale_e) * x
In these tests we feed 'logits' directly as the block input; with no ffn_gate_inp,
router logits == input, so the same tensor plays the role of x and routing logits.
"""
# ===== Helper =====
class FakeExperts(torch.nn.Module):
    """
    Test expert: expert_e(x) = scale_e * x
    """

    def __init__(self, scales):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))

    def forward(self, h, top_k_experts, expert_gate):
        # h: (N, D), top_k_experts: (N, K), expert_gate: (N, K)
        # effective_scale_n = Σ_j p_{n,j} * scale_{expert_idx_{n,j}}
        chosen_scales = self.scales[top_k_experts]  # (N,K)
        eff_scale = (expert_gate * chosen_scales).sum(dim=1)  # (N,)
        return eff_scale.unsqueeze(-1) * h


def make_theta_linear_scales(
    scales, feat_dim, bias_gate=None, bias_up=None, bias_down=None
):
    """
    create minimal theta with per-expert down projection = scale * I and gate/up =I
    """
    n_expt = len(scales)
    ident = torch.eye(feat_dim, dtype=torch.float32)
    ffn_gate = ident.repeat(n_expt, 1, 1)
    ffn_up = ident.repeat(n_expt, 1, 1)
    ffn_down = torch.stack([torch.eye(feat_dim) * s for s in scales], dim=0)
    theta_dict = {
        "ffn_gate.weight": DefaultPrimitiveTensor(
            name="ffn_gate.weight", data=ffn_gate
        ),
        "ffn_up.weight": DefaultPrimitiveTensor(name="ffn_up.weight", data=ffn_up),
        "ffn_down.weight": DefaultPrimitiveTensor(
            name="ffn_down.weight", data=ffn_down
        ),
        "ffn_gate_exps.weight": DefaultPrimitiveTensor(
            name="ffn_gate_exps.weight", data=ffn_gate
        ),
        "ffn_up_exps.weight": DefaultPrimitiveTensor(
            name="ffn_up_exps.weight", data=ffn_up
        ),
        "ffn_down_exps.weight": DefaultPrimitiveTensor(
            name="ffn_down_exps.weight", data=ffn_down
        ),
    }
    # Add bias terms if provided
    if bias_gate is not None:
        if bias_gate.dim() == 1:
            bias_gate = bias_gate.unsqueeze(-1).expand(n_expt, feat_dim)
        theta_dict["ffn_gate.bias"] = DefaultPrimitiveTensor(
            name="ffn_gate.bias", data=bias_gate
        )
        theta_dict["ffn_gate_exps.bias"] = DefaultPrimitiveTensor(
            name="ffn_gate_exps.bias", data=bias_gate
        )

    if bias_up is not None:
        if bias_up.dim() == 1:
            bias_up = bias_up.unsqueeze(-1).expand(n_expt, feat_dim)
        theta_dict["ffn_up.bias"] = DefaultPrimitiveTensor(
            name="ffn_up.bias", data=bias_up
        )
        theta_dict["ffn_up_exps.bias"] = DefaultPrimitiveTensor(
            name="ffn_up_exps.bias", data=bias_up
        )

    if bias_down is not None:
        if bias_down.dim() == 1:
            bias_down = bias_down.unsqueeze(-1).expand(n_expt, feat_dim)
        theta_dict["ffn_down.bias"] = DefaultPrimitiveTensor(
            name="ffn_down.bias", data=bias_down
        )
        theta_dict["ffn_down_exps.bias"] = DefaultPrimitiveTensor(
            name="ffn_down_exps.bias", data=bias_down
        )

    return Theta(theta_dict)


def make_moe_block(
    *,
    expert_count: int,
    expert_used_count: int,
    fake_scales: list[float],
    feature_dim: int,
    normalize_experts=True,
):
    # Build a minimal Theta; we won't use routed FFN theta because we inject FakeExperts
    # ffn_gate_inp is absent => Identity, so router_logits = ffn_input.
    theta = make_theta_linear_scales(fake_scales, feature_dim)
    block = MoeBlock(
        theta=theta,
        rms_epsilon=1e-6,
        moe_activation=torch.nn.functional.silu,
        experts_ffn_moe_block=FakeExperts(fake_scales),
        score_experts=lambda x: torch.softmax(x, dim=-1),
        normalize_experts=normalize_experts,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        expert_shared_count=None,
        n_expert_groups=None,
        n_limited_groups=None,
        route_scale=None,
        model_arch=None,
    )
    return block


def _expected_from_logits(
    logits: torch.Tensor, scales: list[float], top_k: int
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    num_experts = probs.size(-1)
    if top_k < num_experts:
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        masked = torch.zeros_like(probs).scatter(1, topk_idx, topk_probs)
        probs = masked / masked.sum(dim=-1, keepdim=True)
    S = torch.tensor(scales, dtype=logits.dtype, device=logits.device)
    eff_scale = (probs * S).sum(dim=-1, keepdim=True)
    return eff_scale * logits


# 1. One-hot (identity / zero/ arbitrary)
@pytest.mark.parametrize(
    "scales,force_idx,label",
    [
        ([0.0, 1.0, 2.0], 1, "Identity"),
        ([0.0, 1.0, 3.0], 0, "Zero"),
        ([2.5, -1.0, 4.0], 2, "Arbitrary"),
    ],
)
def test_moe_one_hot_variants(deterministic_random_seed, scales, force_idx, label):
    """
    One-hot routing (top_k = 1):
        p ≈ one_hot(j) with j = force_idx  ⇒  y ≈ scale_j * x
    Special cases:
        Identity: scale_j = 1 → y = x
        Zero:     scale_j = 0 → y = 0
        Arbitrary: y = scale_j * x
    """
    num_experts = len(scales)
    top_k = 1
    logits = torch.full((5, num_experts), -80.0)
    logits[:, force_idx] = 80.0
    block = make_moe_block(
        expert_count=num_experts,
        expert_used_count=top_k,
        fake_scales=scales,
        feature_dim=num_experts,
    )

    out = block(logits.unsqueeze(0)).squeeze(0)
    expected = _expected_from_logits(logits, scales, top_k)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6, msg=label)


# 2. Mixture modes(uniform, dense softmax, topk mask)
@pytest.mark.parametrize(
    "mode,scales,top_k,logit_fn",
    [
        ("uniform", [1.0, 2.0, 3.0], None, lambda N, E: torch.zeros(N, E)),
        ("dense_random", [0.0, 1.0, 2.0, 3.0], None, lambda N, E: torch.randn(N, E)),
        ("topk_random", [5.0, -1.0, 2.0, 4.0, -2.0], 2, lambda N, E: torch.randn(N, E)),
    ],
)
def test_moe_mixture_modes(deterministic_random_seed, mode, scales, top_k, logit_fn):
    """
    Mixture modes:
      Uniform (all logits=0, k=E): p_e = 1/E ⇒ y = mean(scales) * x
      Dense random (k=E): y = (Σ_e p_e scale_e) * x
      Top-k random (k < E):
          Let p' = softmax(logits); T = top-k indices of p'
          p_e = p'_e / Σ_{j∈T} p'_j if e ∈ T else 0
          y = (Σ_{e∈T} p_e scale_e) * x
    """
    num_tokens = 6
    num_experts = len(scales)
    logits = logit_fn(num_tokens, num_experts)
    use_k = num_experts if top_k is None else top_k

    block = make_moe_block(
        expert_count=num_experts,
        expert_used_count=use_k,
        fake_scales=scales,
        feature_dim=num_experts,
    )
    out = block(logits.unsqueeze(0)).squeeze(0)
    expected = _expected_from_logits(logits, scales, use_k)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5, msg=mode)


class FakeExpertsSquare(torch.nn.Module):
    """
    Test expert for PreGather parity:
        With W_gate=I, W_up=I, activation=Identity, W_down=scale_e * I
        PreGather expert output = scale_e * (x * x) (elementwise square).
    """

    def __init__(self, scales):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))

    def forward(self, h, top_k_experts, expert_gate):
        chosen_scales = self.scales[top_k_experts]  # (N,K)
        eff_scale = (expert_gate * chosen_scales).sum(dim=1)  # (N,)
        return eff_scale.unsqueeze(-1) * (h * h)


def test_moe_pregather_vs_fake_linear_scales(deterministic_random_seed):
    """
    Parity: PreGatherFFNMOE (producing scale_e * (x*x)) vs FakeExpertsSquare.
    Ensures routing weights and scale aggregation match between implementations.
    """
    from amdsharktank.layers.mixture_of_experts_block import PreGatherFFNMOE  # fallback

    num_expt, top_k, feature_dim = 4, 2, 4
    scales = [0.0, 1.0, 2.0, -1.0]

    # Theta with scales for PreGather path
    theta_pg = make_theta_linear_scales(scales, feature_dim)
    theta_fake = make_theta_linear_scales([1.0] * num_expt, feature_dim)
    pregather = PreGatherFFNMOE(theta_pg, activation_fn=torch.nn.Identity())

    block_pg = MoeBlock(
        theta=theta_pg,
        rms_epsilon=1e-6,
        experts_ffn_moe_block=pregather,
        score_experts=torch.nn.functional.softmax,
        normalize_experts=True,
        expert_count=num_expt,
        expert_used_count=top_k,
    )
    block_fake = MoeBlock(
        theta=theta_fake,
        rms_epsilon=1.0e-6,
        experts_ffn_moe_block=FakeExpertsSquare(scales),
        score_experts=torch.nn.functional.softmax,
        normalize_experts=True,
        expert_count=num_expt,
        expert_used_count=top_k,
    )

    x = torch.randn(6, feature_dim)
    y_pg = block_pg(x.unsqueeze(0)).squeeze(0)
    y_fake = block_fake(x.unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(y_pg, y_fake, atol=1e-6, rtol=1e-6)


def test_pregather_dense_parity_minimal():
    """
    W_gate_e = I, W_up_e = I, W_down_e = scale_e * I
       ⇒ h_e = act(h) ⊙ h   (independent of e)
       ⇒ y = ( Σ_e p_e scale_e ) * ( act(h) ⊙ h )

    We choose top_k = num_experts so all experts participate (no routing sparsity).
    Router logits = input (ffn_gate_inp = Identity), so p = softmax(x_last_dim).
    """
    # Tiny case
    scales = [1.0, -0.5, 2.0]
    feat = 3
    num_experts = 3
    top_k = 3
    theta = make_theta_linear_scales(scales, feat)
    block_preg = MoeBlock(
        theta=theta,
        expert_count=num_experts,
        expert_used_count=top_k,
        experts_ffn_moe_block="PreGatherFFNMOE",
        score_experts=lambda t: torch.softmax(t, dim=-1),
        moe_activation=torch.nn.functional.silu,
        rms_epsilon=1e-6,
    )
    block_dense = MoeBlock(
        theta=theta,
        expert_count=num_experts,
        expert_used_count=top_k,
        experts_ffn_moe_block="DenseFFNMOE",
        score_experts=lambda t: torch.softmax(t, dim=-1),
        moe_activation=torch.nn.functional.silu,
        rms_epsilon=1e-6,
    )
    x_token = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    x = x_token.unsqueeze(0)
    torch.testing.assert_close(block_preg(x), block_dense(x), atol=1e-6, rtol=1e-6)


def test_pregather_analytical_linear_scales():
    """
    Pure closed-form golden test (no helper):
      W_gate=I, W_up=I, W_down_e = scale_e * I, activation=silu, no bias.
      Output per token: ( Σ_j gate_{n,j} * scale_{e_{n,j}} ) * (silu(h_n) ⊙ h_n)
    """
    scales = [0.0, 1.5, -2.0]
    feat = 3
    theta = make_theta_linear_scales(scales, feat)
    layer = PreGatherFFNMOE(theta, activation_fn=torch.nn.functional.silu)
    # 2 tokens
    h = torch.tensor([[0.4, -0.2, 0.3], [-0.5, 0.1, 0.25]], dtype=torch.float32)
    # choose experts per token
    experts = torch.tensor([[1, 2], [0, 2]], dtype=torch.long)
    gate = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
    # manual spec
    base = torch.nn.functional.silu(h) * h
    s_w = torch.tensor(scales)
    eff_scale = torch.stack(
        [
            0.7 * s_w[1] + 0.3 * s_w[2],
            0.4 * s_w[0] + 0.6 * s_w[2],
        ]
    ).unsqueeze(-1)
    ref = eff_scale * base
    out = layer(h, experts, gate)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "llama4,activation,label",
    [
        (False, "silu", "silu_plain"),
        (True, "silu", "silu_llama4"),
        (False, "swiglu", "swiglu_plain"),
        (True, "swiglu", "swiglu_llama4"),
    ],
)
def test_pregather_minimal_closed_form_llama4_swiglu(llama4, activation, label):
    """
    Minimal closed-form golden tests for PreGatherFFNMOE with:
      W_gate = I, W_up = I, W_down_e = scale_e * I, no bias.
    Formulas:
      non-llama4:  y = (Σ_j p_j * scale_{e_j}) * Base(h)
      llama4:      y = (Σ_j p_j^2 * scale_{e_j}) * Base(h)
    Base(h):
      silu  -> silu(h) * h
      swiglu -> ops.swiglu(h)   (requires even last dim)
    """
    scales = [0.0, 1.5, -2.0]
    # Need even feature dim for swiglu.
    feat = 4 if activation == "swiglu" else 3
    theta = make_theta_linear_scales(scales, feat)

    act_fn = torch.nn.functional.silu if activation == "silu" else swiglu
    use_moe_swiglu = False
    if act_fn == swiglu:
        use_moe_swiglu = True
    layer = PreGatherFFNMOE(
        theta,
        activation_fn=act_fn,
        model_arch=("llama4" if llama4 else None),
        use_moe_swiglu=use_moe_swiglu,
    )

    # Tokens (T=2), choose K=2 experts
    if activation == "swiglu":
        # Even dimension example for swiglu
        h = torch.tensor(
            [[0.4, -0.2, 0.3, -0.1], [-0.5, 0.1, 0.25, 0.05]], dtype=torch.float32
        )

    else:
        h = torch.tensor([[0.4, -0.2, 0.3], [-0.5, 0.1, 0.25]], dtype=torch.float32)

    experts = torch.tensor([[1, 2], [0, 2]], dtype=torch.long)
    gate = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)

    # Base activation term
    if activation == "silu":
        base = torch.nn.functional.silu(h) * h
    elif activation == "swiglu":
        base = swiglu(cat([h, h], dim=-1))
    else:
        raise ValueError(f"Unknown activation {activation}")

    S = torch.tensor(scales, dtype=h.dtype)

    if not llama4:
        # coeff = Σ_j p_j * scale_{e_j}
        eff = torch.stack(
            [
                gate[0, 0] * S[experts[0, 0]] + gate[0, 1] * S[experts[0, 1]],
                gate[1, 0] * S[experts[1, 0]] + gate[1, 1] * S[experts[1, 1]],
            ]
        )
    else:
        # llama4 semantics (current implementation): p_j applied inside and outside -> p_j^2
        eff = torch.stack(
            [
                (gate[0, 0] * gate[0, 0]) * S[experts[0, 0]]
                + (gate[0, 1] * gate[0, 1]) * S[experts[0, 1]],
                (gate[1, 0] * gate[1, 0]) * S[experts[1, 0]]
                + (gate[1, 1] * gate[1, 1]) * S[experts[1, 1]],
            ]
        )
    ref = eff.unsqueeze(-1) * base

    out = layer(h, experts, gate)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_pregather_with_bias():
    """
    Test PreGatherFFNMOE._broadcast_bias method for correct bias selection.

    Tests all three bias types (gate, up, down) with expert selection pattern:
    - Given bias = [b0, b1, b2] and experts = [[0, 1], [1, 2]]
    - Token 0 should get biases [b0, b1]
    - Token 1 should get biases [b1, b2]

    The _broadcast_bias method creates one-hot encodings for selected experts,
    then uses matrix multiplication to extract the correct bias values.
    """
    scales = [1.0, 1.0, 1.0]
    feat = 4

    bias_gate = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    bias_up = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
    bias_down = torch.tensor([1000.0, 2000.0, 3000.0], dtype=torch.float32)

    theta = make_theta_linear_scales(
        scales, feat, bias_gate=bias_gate, bias_up=bias_up, bias_down=bias_down
    )
    layer = PreGatherFFNMOE(theta, activation_fn=torch.nn.functional.silu)

    experts = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    # Test gate bias
    bias_gate_expanded = bias_gate.unsqueeze(-1).expand(3, feat)
    broadcasted_gate = layer._broadcast_bias(bias_gate_expanded, experts)
    expected_gate = torch.stack(
        [
            torch.stack([bias_gate_expanded[0], bias_gate_expanded[1]]),
            torch.stack([bias_gate_expanded[1], bias_gate_expanded[2]]),
        ]
    )
    torch.testing.assert_close(broadcasted_gate, expected_gate, atol=1e-6, rtol=1e-6)

    # Test up bias
    bias_up_expanded = bias_up.unsqueeze(-1).expand(3, feat)
    broadcasted_up = layer._broadcast_bias(bias_up_expanded, experts)
    expected_up = torch.stack(
        [
            torch.stack([bias_up_expanded[0], bias_up_expanded[1]]),
            torch.stack([bias_up_expanded[1], bias_up_expanded[2]]),
        ]
    )
    torch.testing.assert_close(broadcasted_up, expected_up, atol=1e-6, rtol=1e-6)

    # Test down bias
    bias_down_expanded = bias_down.unsqueeze(-1).expand(3, feat)
    broadcasted_down = layer._broadcast_bias(bias_down_expanded, experts)
    expected_down = torch.stack(
        [
            torch.stack([bias_down_expanded[0], bias_down_expanded[1]]),
            torch.stack([bias_down_expanded[1], bias_down_expanded[2]]),
        ]
    )
    torch.testing.assert_close(broadcasted_down, expected_down, atol=1e-6, rtol=1e-6)


def test_pregather_with_bias_analytical():
    """
    End-to-end test with hand-calculated expected values for PreGatherFFNMOE with biases.

    Setup: Identity weight matrices (W_gate=I, W_up=I) with scaled down projection
    (W_down_e = scale_e * I) and bias terms. Uses SiLU activation.

    Forward pass calculation:
    1. gate_proj = h + bias_gate[expert_id] (broadcast per feature)
    2. up_proj = h + bias_up[expert_id] (broadcast per feature)
    3. hidden = silu(gate_proj) * up_proj
    4. down = scale[expert_id] * hidden + bias_down[expert_id]
    5. output = weighted sum over selected experts

    Verifies the complete forward pass matches hand calculation.
    """
    scales = [1.0, 2.0]
    feat = 2

    bias_gate = torch.tensor([0.1, 0.2], dtype=torch.float32)
    bias_up = torch.tensor([0.5, 1.0], dtype=torch.float32)
    bias_down = torch.tensor([10.0, 20.0], dtype=torch.float32)

    theta = make_theta_linear_scales(
        scales, feat, bias_gate=bias_gate, bias_up=bias_up, bias_down=bias_down
    )
    layer = PreGatherFFNMOE(theta, activation_fn=torch.nn.functional.silu)

    h = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    experts = torch.tensor([[0, 1]], dtype=torch.long)
    gate_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

    # Manual calculation for expert 0
    gate_proj_0 = h + bias_gate[0].item()
    up_proj_0 = h + bias_up[0].item()
    hidden_0 = torch.nn.functional.silu(gate_proj_0) * up_proj_0
    down_0 = scales[0] * hidden_0 + bias_down[0].item()

    # Manual calculation for expert 1
    gate_proj_1 = h + bias_gate[1].item()
    up_proj_1 = h + bias_up[1].item()
    hidden_1 = torch.nn.functional.silu(gate_proj_1) * up_proj_1
    down_1 = scales[1] * hidden_1 + bias_down[1].item()

    # Weighted combination
    expected = gate_weights[0, 0] * down_0 + gate_weights[0, 1] * down_1

    out = layer(h, experts, gate_weights)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
