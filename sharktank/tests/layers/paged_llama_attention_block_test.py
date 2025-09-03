# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from sharktank.layers.configs.llm_configs import LlamaHParams, LlamaModelConfig
from sharktank.layers.paged_attention import CacheAllocation
import unittest
import torch
from iree.turbine import aot
from sharktank.layers import (
    PagedLlamaAttentionBlock,
    PagedAttention,
    build_rotary_layer,
)
from sharktank.layers.testing import make_llama_attention_block_theta
from sharktank.types.tensors import DefaultPrimitiveTensor

from transformers import LlamaConfig
import pytest
import math
import os
from pathlib import Path
from sharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    iree_to_torch,
)
from sharktank.utils.export import export_model_mlir
from sharktank.utils.testing import TempDirTestBase
import iree.compiler
from iree.turbine.aot import (
    FxProgramsBuilder,
    export as export_fx_programs,
)
from parameterized import parameterized
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

torch.manual_seed(123456)


class PagedLlamaAttentionBlockTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.transformer_block_count = 13
        self.block_index = 1
        self.shard_count = 3
        self.head_count_kv = 2 * self.shard_count
        self.attention_head_count = 5 * self.head_count_kv
        self.attention_head_dim = 11 * 2
        self.rms_epsilon = 0.01
        self.block_seq_stride = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.embedding_length = self.attention_head_count * self.attention_head_dim
        self.rope_dimension_count = self.attention_head_dim
        self.block_seqlen = 7
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = None
        self.batch_size = 3
        self.start_index = 0

    @pytest.mark.xfail(
        torch.__version__ >= (2, 4),
        reason="https://github.com/nod-ai/shark-ai/issues/684",
    )
    @pytest.mark.skipif(
        torch.__version__ >= (2, 5),
        reason="https://github.com/nod-ai/shark-ai/issues/684, error slows down CI",
    )
    def testExportNondecomposed(self):
        dtype = torch.float32

        theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )

        hp = LlamaHParams(
            model_arch="llama",
            context_length=self.max_seqlen,
            embedding_length=self.embedding_length,
            block_count=self.transformer_block_count,
            feed_forward_length=None,
            attention_head_count=self.attention_head_count,
            attention_head_count_kv=self.head_count_kv,
            attn_head_dim=self.attention_head_dim,
            attention_layer_norm_rms_epsilon=self.rms_epsilon,
        )
        config = LlamaModelConfig(
            hp,
            kv_cache_dtype=dtype,
            attention_dtype=dtype,
            block_seq_stride=self.block_seq_stride,
        )

        attn = PagedLlamaAttentionBlock(
            theta=theta,
            config=config,
            model_arch="llama",
            block_index=self.block_index,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
            attention_kernel="torch",
        )

        cache_state = attn.paged_attention.allocate(self.page_count)
        cache_state.allocation[0] = torch.rand(cache_state[0].shape, dtype=dtype)

        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )

        embedding_module = build_rotary_layer(
            rope_dimension_count=self.rope_dimension_count,
            rope_freq_base=self.rope_freq_base,
        )

        class MyModule(torch.nn.Module):
            def forward(self, h, seq_block_ids, cache_state: list[torch.Tensor]):
                cache_state = CacheAllocation(cache_state)
                return attn.forward(
                    h,
                    embedding=embedding_module,
                    seq_block_ids=seq_block_ids,
                    cache_state=cache_state,
                )

        mod = MyModule()
        h = torch.rand(
            [
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ]
        )
        mod.forward(h, seq_block_ids, cache_state.allocation)
        ep = torch.export.export(
            mod,
            args=(
                h,
                seq_block_ids,
                cache_state.allocation,
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("torch.aten._scaled_dot_product_flash_attention_for_cpu", asm)


# === Sink attention test ===

# Shapes: (bs, seq_len, n_heads, kv_heads, head_dim)
_SHAPE_CASES = [
    (1, 64, 8, 1, 64),
    (2, 128, 8, 2, 64),
]
_CONTEXT_LEN = [2048]
_DT_CASES = [
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 2e-3, 1e-3),
    (torch.bfloat16, 2e-2, 1e-2),
]
_MODES = ["prefill", "decode"]

_SINK_CASES = [  # sliding_window, sink_scale
    (None, None),  # base path
    (19, 0.25),  # sink path enabled
]


def _reference_sink_batched(q, k, v, sink, mode, sliding_window):

    # replicate k and v for GQA
    bs, n_tokens, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    assert (
        n_heads % n_kv_heads == 0
    ), "num_attention_heads must be a multiple of num_kv_heads"
    q_groups = n_heads // n_kv_heads  # Q (query groups per kv head)
    q_mul = q_groups
    q = q.reshape(bs, n_tokens, n_kv_heads, q_mul, head_dim)

    k_ = k.unsqueeze(3).expand(-1, -1, -1, q_mul, -1)
    v_ = v.unsqueeze(3).expand(-1, -1, -1, q_mul, -1)

    sm_scale = 1.0 / math.sqrt(head_dim)
    q_ = q

    sink_ = sink.reshape(n_kv_heads, q_mul, 1, 1).expand(-1, -1, n_tokens, -1)
    sink_ = sink_.unsqueeze(0).expand(bs, -1, -1, -1, -1)

    mask = torch.triu(q_.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    qk_ = torch.einsum("bqhmd,bkhmd->bhmqk", q_, k_) * sm_scale
    qk_ = qk_ + mask[None, None, :, :]

    qk_ = torch.cat([qk_, sink_], dim=-1)
    w = torch.softmax(qk_, dim=-1)[..., :-1]  # drop sink column

    attn = torch.einsum("bhmqk,bkhmd->bqhmd", w, v_)

    out = attn.reshape(bs, n_tokens, n_kv_heads * q_mul, -1).permute(0, 2, 1, 3)
    if mode == "decode":
        out = out[:, :, -1:, :]
    return out


def _reference_base(q, k, v, mode):
    bs, n_tokens, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]

    if n_kv_heads != n_heads:
        assert (
            n_heads % n_kv_heads == 0
        ), "num_attention_heads must be a multiple of num_kv_heads"
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    # Causal mask
    mask = torch.triu(q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # Permute to (B,H,T,D) for SDPA
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)

    out = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)

    if mode == "decode":
        out = out[:, :, -1:, :]  # keep last token

    return out


def _make_reference_for_case(q, k, v, mode, sliding_window, sink):
    # Choose the correct reference implementation for this configuration.
    if (sliding_window is not None) and (sink is not None):
        return _reference_sink_batched(q, k, v, sink, mode, sliding_window)
    else:
        return _reference_base(q, k, v, mode)


def _create_sink_tensor(total_q_heads, dtype, sink_scale, sink_size=1):
    # total_q_heads == Hq == num_attention_heads
    if sink_scale is None:
        return None
    return torch.full((sink_size, total_q_heads), sink_scale, dtype=dtype)


def _make_qkv(bs, seqlen, n_heads, kv_heads, head_dim, dtype):
    """
    q: 4D (B,T,Hq,D)
    k,v: 4D (B,T,Hkv,D)
    """
    q = torch.randn(bs, seqlen, n_heads, head_dim, dtype=dtype)
    k = torch.randn(bs, seqlen, kv_heads, head_dim, dtype=dtype)
    v = torch.randn(bs, seqlen, kv_heads, head_dim, dtype=dtype)
    return q, k, v


def decode_attention_mask(seq_lens, batch_seqlen, attention_dtype, device):
    range_vector = torch.arange(0, batch_seqlen, 1, device=device)
    matrix = seq_lens.unsqueeze(dim=-1)
    mask = range_vector >= matrix
    dtype = (
        torch.float32 if attention_dtype == torch.float8_e4m3fnuz else attention_dtype
    )
    numeric_mask = torch.where(
        mask, torch.tensor(float("-inf"), dtype=dtype, device=device), 0
    ).to(dtype)
    return numeric_mask.unsqueeze(1).unsqueeze(1).to(device)


class PrefillWrapperEager(torch.nn.Module):
    def __init__(
        self,
        pa: PagedAttention,
        block_index: int,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor | None,
    ):
        super().__init__()
        self.pa = pa
        self.block_index = block_index
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        if sink is not None:
            self.register_buffer("sink", sink)
        else:
            self.sink = None

    def forward(self, q, k, v, cache_state, seq_block_ids, mask=None):
        fn_or_result = self.pa.forward_prefill(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=self.block_index,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
            mask=mask,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )
        return fn_or_result


class PrefillAndDecodeWrapper(torch.nn.Module):
    """Combined prefill (full sequence) + decode last token in one call."""

    def __init__(
        self,
        pa: PagedAttention,
        block_index: int,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor | None,
    ):
        super().__init__()
        self.pa = pa
        self.block_index = block_index
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        if sink is not None:
            self.register_buffer("sink", sink)
        else:
            self.sink = None

    def forward(self, q, k, v, cache_state, seq_block_ids, start_positions, mask):
        _ = self.pa.forward_prefill(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=self.block_index,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
            mask=None,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )
        q_last = q[:, -1:, ...]
        k_last = k[:, -1:, ...]
        v_last = v[:, -1:, ...]
        return self.pa.forward_decode(
            q=q_last,
            k=k_last,
            v=v_last,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=self.block_index,
            start_positions=start_positions,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            mask=mask,
            scale=None,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )


def _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype):
    bs, seq_len, n_heads = q.shape[:3]
    stride = pa.block_seq_stride

    blocks = math.ceil(seq_len / stride)
    # batch b uses pages [b*blocks, (b+1)*blocks).
    per_batch_offset = (
        torch.arange(bs, device=q.device, dtype=torch.int64)[:, None] * blocks
    )
    seq_block_ids = (
        per_batch_offset
        + torch.arange(blocks, device=q.device, dtype=torch.int64)[None, :]
    )
    cache_state = pa.allocate(
        page_count=context_len // stride,
    )

    if mode == "prefill":
        wrapper = PrefillWrapperEager(pa, 0, n_heads, sliding_window, sink)
        prefill = wrapper(q, k, v, cache_state, seq_block_ids)
        return prefill

    else:
        past_len = seq_len - 1
        start_positions = torch.full((bs,), past_len, device=q.device, dtype=torch.long)
        seq_lens = torch.full((bs,), seq_len, device=q.device, dtype=torch.long)

        decode_mask = decode_attention_mask(
            seq_lens,
            seq_block_ids.shape[1] * pa.block_seq_stride,
            dtype,
            q.device,
        ).to(q.device)

        wrapper = PrefillAndDecodeWrapper(pa, 0, n_heads, sliding_window, sink)
        out = wrapper(
            q,
            k,
            v,
            cache_state,
            seq_block_ids,
            start_positions,
            mask=decode_mask,
        )

        return out


class TestPagedAttentionForwardSinkEager:
    @pytest.mark.parametrize(("dtype", "atol", "rtol"), _DT_CASES)
    @pytest.mark.parametrize(("sliding_window", "sink_scale"), _SINK_CASES)
    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize(
        ("bs", "seqlen", "n_heads", "kv_heads", "head_dim"), _SHAPE_CASES
    )
    @pytest.mark.parametrize("context_len", _CONTEXT_LEN)
    def test_forward_sink_eager(
        self,
        dtype,
        atol,
        rtol,
        sliding_window,
        mode,
        bs,
        seqlen,
        n_heads,
        kv_heads,
        head_dim,
        sink_scale,
        context_len,
    ):
        torch.manual_seed(1234)
        pa = PagedAttention(
            transformer_block_count=1,
            attn_head_count=kv_heads,
            attn_head_dim=head_dim,
            attn_type="gqa",
            cache_partition_count=2,
            block_seq_stride=16,
            cache_dtype=dtype,
            attn_dtype=dtype,
            device=None,
        )

        sink = _create_sink_tensor(n_heads, dtype, sink_scale)
        q, k, v = _make_qkv(bs, seqlen, n_heads, kv_heads, head_dim, dtype)

        ref = _make_reference_for_case(q, k, v, mode, sliding_window, sink).to(q.dtype)
        out = _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype)

        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# === IREE vs Eager sink attention test ===
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


def _build_fx_program_for_mode(
    pa: PagedAttention,
    n_heads: int,
    sliding_window: int | None,
    sink: torch.Tensor | None,
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_state: tuple,
    seq_block_ids: torch.Tensor,
    start_positions: torch.Tensor | None = None,
    decode_mask: torch.Tensor | None = None,
):
    """Returns an FxProgramsBuilder with the appropriate exported program for the mode."""
    if mode == "prefill":
        prefill_wrapper = PrefillWrapperEager(
            pa, 0, n_heads, sliding_window, sink
        ).eval()
        fxb = FxProgramsBuilder(prefill_wrapper)

        @fxb.export_program(
            name="paged_attn_sink_prefill",
            args=(
                q.clone(),
                k.clone(),
                v.clone(),
                cache_state.allocation,
                seq_block_ids.clone(),
            ),
            dynamic_shapes=None,
            strict=False,
        )
        def _(m, q_, k_, v_, cache_alloc_, seq_block_ids_):
            from sharktank.layers.paged_attention import CacheAllocation

            cache_state_ = CacheAllocation(cache_alloc_)
            return m(q_, k_, v_, cache_state_, seq_block_ids_)

        return fxb
    # decode
    decode_wrapper = PrefillAndDecodeWrapper(
        pa, 0, n_heads, sliding_window, sink
    ).eval()
    fxb = FxProgramsBuilder(decode_wrapper)

    @fxb.export_program(
        name="paged_attn_sink_decode",
        args=(
            q.clone(),
            k.clone(),
            v.clone(),
            cache_state.allocation,
            seq_block_ids.clone(),
            start_positions.clone(),
            decode_mask.clone(),
        ),
        dynamic_shapes=None,
        strict=False,
    )
    def _(m, ql_, kl_, vl_, cache_alloc_, seq_block_ids_, start_pos_, mask_):
        from sharktank.layers.paged_attention import CacheAllocation

        cache_state_ = CacheAllocation(cache_alloc_)
        return m(ql_, kl_, vl_, cache_state_, seq_block_ids_, start_pos_, mask_)

    return fxb


@pytest.mark.usefixtures("iree_flags", "device")
class TestPagedAttentionForwardSinkIree(TempDirTestBase):
    """Test PagedAttention forward with sink tensor in IREE."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

    @parameterized.expand(
        [
            (
                dt,
                atol,
                rtol,
                mode,
                bs,
                seqlen,
                n_heads,
                kv_heads,
                head_dim,
                sink_scale,
                sliding_window,
                context_len,
            )
            for (dt, atol, rtol) in _DT_CASES
            for mode in _MODES
            for (bs, seqlen, n_heads, kv_heads, head_dim) in _SHAPE_CASES
            for (sliding_window, sink_scale) in _SINK_CASES
            for context_len in _CONTEXT_LEN
        ]
    )
    def test_forward_sink_iree(
        self,
        dtype,
        atol,
        rtol,
        mode,
        bs,
        seqlen,
        n_heads,
        kv_heads,
        head_dim,
        sink_scale,
        sliding_window,
        context_len,
    ):
        driver_env = getattr(self, "iree_hal_target_device", None)
        driver, compile_args, cpu_like = _resolve_iree_compile(driver_env)
        if cpu_like and dtype is torch.bfloat16:
            pytest.xfail(
                "llvm-cpu lacks bf16 runtime builtins (__truncsfbf2); run bf16 on GPU-only."
            )
        logger.info(
            "Testing PagedAttention forward with sink tensor in IREE. "
            f"bs={bs}, seqlen={seqlen}, n_heads={n_heads}, n_kv_heads={kv_heads}, head_dim={head_dim}, "
            f"sink_scale={sink_scale}, sliding_window={sliding_window}, "
            f"context_len={context_len}, "
            f"mode={mode}, driver={driver}, datatype={dtype}"
        )
        pa = PagedAttention(
            transformer_block_count=1,
            attn_head_count=kv_heads,
            attn_head_dim=head_dim,
            attn_type="gqa",
            cache_partition_count=2,
            block_seq_stride=16,
            cache_dtype=dtype,
            attn_dtype=dtype,
            device=None,
        )
        sink = _create_sink_tensor(n_heads, dtype, sink_scale)
        q, k, v = _make_qkv(bs, seqlen, n_heads, kv_heads, head_dim, dtype)

        expected = _run_pa_eager(
            pa, mode, q, k, v, sink, sliding_window, context_len, dtype
        )

        # Build inputs for compile
        stride = pa.block_seq_stride
        blocks = math.ceil(seqlen / stride)
        per_batch_offset = (
            torch.arange(bs, device=q.device, dtype=torch.int64)[:, None] * blocks
        )
        seq_block_ids = (
            per_batch_offset
            + torch.arange(blocks, device=q.device, dtype=torch.int64)[None, :]
        )
        cache_state = pa.allocate(page_count=context_len // stride)
        past_len = seqlen - 1
        start_positions = torch.full((bs,), past_len, device=q.device, dtype=torch.long)
        seq_lens = torch.full((bs,), seqlen, device=q.device, dtype=torch.long)
        decode_mask = decode_attention_mask(
            seq_lens,
            seq_block_ids.shape[1] * pa.block_seq_stride,
            dtype,
            q.device,
        ).to(q.device)

        fxb = _build_fx_program_for_mode(
            pa=pa,
            n_heads=n_heads,
            sliding_window=sliding_window,
            sink=sink,
            mode=mode,
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            start_positions=start_positions if mode == "decode" else None,
            decode_mask=decode_mask if mode == "decode" else None,
        )

        # Compile
        mlir_path = self._temp_dir / "paged_sink.mlir"
        vmfb_path = self._temp_dir / "paged_sink.vmfb"
        export_fx_programs(fxb).save_mlir(mlir_path)
        logger.info("Saved MLIR to %s", mlir_path.resolve())

        iree.compiler.compile_file(
            str(mlir_path), output_file=str(vmfb_path), extra_args=compile_args
        )
        logger.info("Saved VMFB to %s", vmfb_path.resolve())

        iree_devices = get_iree_devices(driver=driver, device_count=1)

        def run_iree_module(devs):
            logger.info("Loading IREE module from %s", vmfb_path.resolve())
            module, vm_ctx, _ = load_iree_module(
                module_path=str(vmfb_path), devices=devs
            )

            if mode == "prefill":
                _args = [q, k, v, cache_state.allocation, seq_block_ids]
                fn = "paged_attn_sink_prefill"
            else:
                _args = [
                    q,
                    k,
                    v,
                    cache_state.allocation,
                    seq_block_ids,
                    start_positions,
                    decode_mask,
                ]
                fn = "paged_attn_sink_decode"

            iree_args = prepare_iree_module_function_args(args=_args, devices=devs)
            logger.info("Invoking function %s", fn)

            iree_result = run_iree_module_function(
                module=module,
                vm_context=vm_ctx,
                args=iree_args,
                device=devs[0],
                function_name=fn,
            )

            return iree_to_torch(*iree_result)[0]

        iree_out = with_iree_device_context(run_iree_module, iree_devices)

        assert iree_out.shape == expected.shape
        torch.testing.assert_close(iree_out, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
