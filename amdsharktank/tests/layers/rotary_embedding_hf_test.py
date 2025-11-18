# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import os
from amdsharktank.layers.rotary_embedding_hf import RotaryEmbeddingLayer
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig
import pytest
import math

from amdsharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    iree_to_torch,
)
from amdsharktank import ops
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.testing import TempDirTestBase, assert_tensor_close
import iree.compiler
from iree.turbine.aot import (
    FxProgramsBuilder,
    export as export_fx_programs,
)
from parameterized import parameterized

logger = get_logger(__name__)


class HFRotaryEmbedding(torch.nn.Module):
    def __init__(self, config, interleaved: bool = True):
        super().__init__()
        self._rotary = LlamaRotaryEmbedding(config=config)
        self.interleaved = interleaved

    def forward(self, q, k, positions):
        cos, sin = self._rotary(q, positions)
        dim = q.shape[-1]
        if self.interleaved:
            q = q.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
            k = k.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        if self.interleaved:
            q = q.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
            k = k.unflatten(-1, (dim // 2, 2)).transpose(-1, -2).flatten(-2, -1)
        return q, k


class STRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim,
        rope_theta,
        use_base_frequency_scaling: bool = False,
        interleaved: bool = True,
        yarn_beta_slow: float | None = None,
        yarn_beta_fast: float | None = None,
        yarn_factor: float | None = None,
        yarn_original_context_len: int | None = None,
    ):
        super().__init__()
        self._rotary = RotaryEmbeddingLayer(
            head_dim=head_dim,
            rope_theta=rope_theta,
            use_base_frequency_scaling=use_base_frequency_scaling,
            interleaved=interleaved,
            yarn_beta_slow=yarn_beta_slow,
            yarn_beta_fast=yarn_beta_fast,
            yarn_factor=yarn_factor,
            yarn_original_context_len=yarn_original_context_len,
        )

    def forward(self, q, k, positions):
        cossin_cache = self._rotary.compute_sincos_cache(positions, q.dtype)
        q = self._rotary(q, cossin_cache)
        k = self._rotary(k, cossin_cache)
        return (q, k)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("prefill_offset", [True, False])
def test_rotary_interweaved(
    deterministic_random_seed, dtype: torch.dtype, prefill_offset: bool
):
    bs = 2
    length = 256
    heads = 16
    dims = 128

    hf_config = LlamaConfig(
        max_position_embeddings=131072,
        rope_theta=500000,
    )

    hf_rotary = HFRotaryEmbedding(hf_config, interleaved=False)

    st_rotary = STRotaryEmbedding(head_dim=dims, rope_theta=500000, interleaved=False)

    def test_prefill():
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)

        start_positions = ops.arange(0, bs)
        positions_seq = ops.arange(0, length)

        if prefill_offset:
            position_ids = positions_seq.unsqueeze(0) + start_positions.unsqueeze(1)
        else:
            position_ids = positions_seq.unsqueeze(0)

        hf_results = hf_rotary(q, k, position_ids)
        st_results = st_rotary(q, k, position_ids)
        assert_tensor_close(hf_results, st_results)

    def test_decode():
        q = torch.randn(bs, 1, heads, dims)
        k = torch.randn(bs, 1, heads, dims)
        position_ids = torch.randint(0, length, (bs, 1))
        hf_results = hf_rotary(q, k, position_ids)
        st_results = st_rotary(q, k, position_ids)
        assert_tensor_close(hf_results, st_results)

    test_prefill()
    test_decode()


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2e-5, 1e-5),
        (torch.float16, None, None),
        (torch.bfloat16, None, None),
    ],
)
def test_rotary_interleaved(
    deterministic_random_seed, dtype: torch.dtype, atol: float, rtol: float
):
    bs = 2
    length = 256
    heads = 16
    dims = 128

    hf_config = LlamaConfig(
        max_position_embeddings=131072,
        rope_theta=500000,
    )

    hf_rotary = HFRotaryEmbedding(hf_config, interleaved=True)

    st_rotary = STRotaryEmbedding(head_dim=dims, rope_theta=500000, interleaved=True)

    # amdsharktank RoPE implementation does permutation along the reduction
    # dimension of Q @ K.T matmul, and is only correct post Q @ K.T matmul.
    # The HF implementation also relies on this, which is why you will notice
    # we do the unflatten + transpose + flatten post hf_rotary application.
    def rot_and_qk(rot, q, k, position_ids):
        q, k = rot(q, k, position_ids)
        q = q.transpose(1, 2).flatten(0, 1)
        k = k.transpose(1, 2).flatten(0, 1)
        out = q @ k.transpose(1, 2)
        return out

    def test_prefill():
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)
        position_ids = ops.arange(0, length)[None, :].repeat(bs, 1)
        leave = rot_and_qk(hf_rotary, q, k, position_ids)
        weave = rot_and_qk(st_rotary, q, k, position_ids)
        # Use a bigger atol because we are doing a matmul.
        assert_tensor_close(leave, weave, atol=atol, rtol=rtol)

    def test_decode():
        q = torch.randn(bs, 1, heads, dims, dtype=dtype)
        k = torch.randn(bs, 1, heads, dims, dtype=dtype)
        position_ids = torch.randint(0, length, (bs, 1))
        leave = rot_and_qk(hf_rotary, q, k, position_ids)
        weave = rot_and_qk(st_rotary, q, k, position_ids)
        # Use a bigger atol because we are doing a matmul.
        assert_tensor_close(leave, weave, atol=atol, rtol=rtol)

    test_prefill()
    test_decode()


# ---------------------------
# OpenWeight reference and tests
# ---------------------------
OPENWEIGHT_CFG = {
    "rope_theta": 150000.0,
    "yarn_factor": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_beta_fast": 32.0,
    "yarn_original_context_len": 4096,
}
_SHAPE_CASES = [
    (1, 128, 8, 64),
    (2, 128, 8, 64),
    (2, 256, 16, 64),
]


def _make_inputs(
    mode: str, bs: int, length: int, heads: int, dims: int, dtype: torch.dtype
):
    if mode == "prefill":
        q = torch.randn(bs, length, heads, dims, dtype=dtype)
        k = torch.randn(bs, length, heads, dims, dtype=dtype)
        position_ids = ops.arange(0, length, device=q.device)[None, :].repeat(bs, 1)
    else:
        q = torch.randn(bs, 1, heads, dims, dtype=dtype)
        k = torch.randn(bs, 1, heads, dims, dtype=dtype)
        position_ids = torch.randint(0, length, (bs, 1), device=q.device)
    return q, k, position_ids


class ReferenceOpenWeightRotary(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(2).to(x.dtype)
        sin = sin.unsqueeze(2).to(x.dtype)
        x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = torch.cat((o1, o2), dim=-1)
        return out

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            ops.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                ops.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin_from_position(self, position_ids: torch.Tensor):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        angles = position_ids[:, :, None].to(torch.float32) * inv_freq[
            None, None, :
        ].to(torch.float32)
        cos = angles.cos() * concentration
        sin = angles.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._compute_cos_sin_from_position(position_ids)

        query = self._apply_rotary_emb(query, cos, sin)

        key = self._apply_rotary_emb(key, cos, sin)

        return query, key


class TestRotaryOpenWeightEager:
    @pytest.mark.parametrize(
        ("dtype", "atol", "rtol"),
        [
            (torch.float32, 3e-5, 1e-5),
            (torch.float16, 2e-3, 1e-3),
            (torch.bfloat16, 2e-3, 1e-3),
        ],
    )
    @pytest.mark.parametrize("mode", ["prefill", "decode"])
    @pytest.mark.parametrize(("bs", "length", "heads", "dims"), _SHAPE_CASES)
    def test_rotary_openweight_interweaved(
        self,
        dtype: torch.dtype,
        atol: float,
        rtol: float,
        mode: str,
        bs: int,
        length: int,
        heads: int,
        dims: int,
    ):

        torch.manual_seed(1234)
        bs, length, heads, dims = bs, length, heads, dims

        st_rotary = STRotaryEmbedding(
            head_dim=dims,
            rope_theta=OPENWEIGHT_CFG["rope_theta"],
            interleaved=False,  # openweight
            use_base_frequency_scaling=True,
            yarn_factor=OPENWEIGHT_CFG["yarn_factor"],
            yarn_beta_slow=OPENWEIGHT_CFG["yarn_beta_slow"],
            yarn_beta_fast=OPENWEIGHT_CFG["yarn_beta_fast"],
            yarn_original_context_len=OPENWEIGHT_CFG["yarn_original_context_len"],
        )

        ref_rotary = ReferenceOpenWeightRotary(
            head_dim=dims,
            base=OPENWEIGHT_CFG["rope_theta"],
            scaling_factor=OPENWEIGHT_CFG["yarn_factor"],
            ntk_alpha=OPENWEIGHT_CFG["yarn_beta_slow"],
            ntk_beta=OPENWEIGHT_CFG["yarn_beta_fast"],
            initial_context_length=OPENWEIGHT_CFG["yarn_original_context_len"],
            dtype=dtype,
        )
        q, k, position_ids = _make_inputs(mode, bs, length, heads, dims, dtype)

        st_q, st_k = st_rotary(q, k, position_ids)
        ref_q, ref_k = ref_rotary(q, k, position_ids)

        assert_tensor_close(st_q, ref_q, atol=atol, rtol=rtol)
        assert_tensor_close(st_k, ref_k, atol=atol, rtol=rtol)


def _resolve_iree_compile(driver_env: str | None):
    # Normalize driver alias from env and map CPU requests to local + llvm-cpu backend.
    requested = (driver_env or os.getenv("IREE_HAL_TARGET_DEVICE") or "hip").lower()
    cpu_aliases = {"llvm-cpu", "cpu", "local"}
    if requested in cpu_aliases:
        runtime_driver = "local-task"
        compile_args = ["--iree-hal-target-backends=llvm-cpu"]
        cpu_like = True
        return runtime_driver, compile_args, cpu_like

    # GPU/backends
    driver = requested
    hip_target = os.getenv("IREE_HIP_TARGET", "gfx942")
    compile_args: list[str] = [f"--iree-hal-target-device={driver}"]
    if driver == "hip":
        compile_args.append(f"--iree-hip-target={hip_target}")
    runtime_driver = driver
    cpu_like = False
    return runtime_driver, compile_args, cpu_like


def _build_st_rotary_eager(dims):
    return STRotaryEmbedding(
        head_dim=dims,
        rope_theta=OPENWEIGHT_CFG["rope_theta"],
        interleaved=False,  # openweight use interweaved
        use_base_frequency_scaling=True,
        yarn_factor=OPENWEIGHT_CFG["yarn_factor"],
        yarn_beta_slow=OPENWEIGHT_CFG["yarn_beta_slow"],
        yarn_beta_fast=OPENWEIGHT_CFG["yarn_beta_fast"],
        yarn_original_context_len=OPENWEIGHT_CFG["yarn_original_context_len"],
    ).eval()


class RotaryWrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, q, k, pos):
        return self.inner(q, k, pos)


@pytest.mark.usefixtures("iree_flags", "device")
class TestRotaryOpenWeightIree(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

    @parameterized.expand(
        [
            (dt, atol, rtol, mode, bs, length, heads, dims)
            for (dt, atol, rtol) in [
                (torch.float32, 1e-4, 1e-5),
                (torch.float16, 2e-3, 1e-3),
                (torch.bfloat16, 2e-2, 1e-2),
            ]
            for mode in ("prefill", "decode")
            for (bs, length, heads, dims) in _SHAPE_CASES
        ]
    )
    def test_rotary_openweight_interweaved_iree(
        self,
        dtype: torch.dtype,
        atol: float,
        rtol: float,
        mode: str,
        bs: int,
        length: int,
        heads: int,
        dims: int,
    ):
        """
        IREE vs eager test (pattern similar to IreeVsEagerLLMTester: eager first,
        then compiled IREE invocation, then compare).
        """
        driver_env = getattr(self, "iree_hal_target_device", None)
        driver, compile_args, cpu_like = _resolve_iree_compile(driver_env)
        if cpu_like and dtype is torch.bfloat16:
            pytest.xfail(
                "llvm-cpu lacks bf16 runtime builtins (__truncsfbf2); run bf16 on GPU-only."
            )

        logger.info(
            "Testing rotary openweight interleaved IREE with "
            f"bs={bs}, length={length}, heads={heads}, dims={dims}, dtype={dtype}, "
            f"mode={mode}, driver={driver}"
        )
        q, k, position_ids = _make_inputs(mode, bs, length, heads, dims, dtype)

        # Eager (reference).
        st_rotary = _build_st_rotary_eager(dims)
        eager_q, eager_k = st_rotary(q, k, position_ids)

        # FX capture
        mlir_path = self._temp_dir / "rotary.mlir"
        vmfb_path = self._temp_dir / "rotary.vmfb"
        wrapper = RotaryWrapper(st_rotary).eval()

        fxb = FxProgramsBuilder(wrapper)

        @fxb.export_program(
            name="rotary_openweight_fw",
            args=(q.clone(), k.clone(), position_ids.clone()),
            dynamic_shapes=None,
            strict=False,
        )
        def _(model, q_, k_, pos_):
            return model(q_, k_, pos_)

        export_fx_programs(fxb).save_mlir(mlir_path)
        logger.info("Saved MLIR to %s", mlir_path.resolve())

        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            extra_args=compile_args,
        )
        logger.info("Saved VMFB to %s", vmfb_path.resolve())

        iree_devices = get_iree_devices(driver=driver, device_count=1)

        def run_iree_module(iree_devices):
            logger.info("Loading IREE module from %s", vmfb_path.resolve())
            iree_module, iree_vm_context, _ = load_iree_module(
                module_path=str(vmfb_path),
                devices=iree_devices,
            )
            iree_args = prepare_iree_module_function_args(
                args=[q, k, position_ids], devices=iree_devices
            )
            logger.info("Invoking function 'rotary_openweight_fw'")
            iree_result = run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                device=iree_devices[0],
                function_name="rotary_openweight_fw",
            )

            return iree_to_torch(*iree_result)

        iree_results = with_iree_device_context(run_iree_module, iree_devices)
        i_q, i_k = iree_results[0], iree_results[1]
        assert (
            i_q.shape == eager_q.shape
        ), f"Q shape mismatch {i_q.shape} vs {eager_q.shape}"
        assert (
            i_k.shape == eager_k.shape
        ), f"K shape mismatch {i_k.shape} vs {eager_k.shape}"
        assert_tensor_close(i_q, eager_q, atol=atol, rtol=rtol)
        assert_tensor_close(i_k, eager_k, atol=atol, rtol=rtol)
