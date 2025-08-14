# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

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

        cache = PagedAttention(
            transformer_block_count=self.transformer_block_count,
            attn_head_count=self.head_count_kv,
            attn_head_dim=self.attention_head_dim,
            cache_partition_count=self.cache_partition_count,
            block_seq_stride=self.block_seq_stride,
            cache_dtype=dtype,
            attn_dtype=dtype,
        )

        cache_state = cache.allocate(self.page_count)
        cache_state[0] = torch.rand(cache_state[0].shape, dtype=dtype)

        theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )
        attn = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=self.block_index,
            cache=cache,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
            attention_kernel="torch",
        )

        seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
            self.batch_size, -1
        )

        embedding_module = build_rotary_layer(
            rope_dimension_count=self.rope_dimension_count,
            max_seqlen=self.max_seqlen,
            rope_freq_base=self.rope_freq_base,
        )

        class MyModule(torch.nn.Module):
            def forward(self, h, seq_block_ids, cache_state):
                return attn.forward(
                    h,
                    seq_block_ids=seq_block_ids,
                    embedding=embedding_module,
                    start_index=0,
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
        mod.forward(h, seq_block_ids, cache_state)
        ep = torch.export.export(
            mod,
            args=(
                h,
                seq_block_ids,
                cache_state,
            ),
        )
        output = aot.export(ep)
        output.verify()
        asm = str(output.mlir_module)
        self.assertIn("torch.aten._scaled_dot_product_flash_attention_for_cpu", asm)


# === Sink attention test ===

# Shapes: (bs, seq_len, n_heads, head_dim)
_SHAPE_CASES = [
    (1, 64, 8, 64),
    (2, 128, 8, 64),
]
_CONTEXT_LEN = [2048]
_DT_CASES = [
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 2e-3, 1e-3),
    (torch.bfloat16, 2e-2, 1e-2),
]
_MODES = ["prefill", "decode"]
_SLIDING_WINDOWS = [0, 1, 19]
_SINK_SCALE = [0, 0.25, 1]
_Q_MUL = [1]


def _reference(q, k, v, sink, mode, sliding_window):
    # q,k,v: (B,T,H,1,D)
    B, T, H, q_mul, D = q.shape
    sm_scale = 1 / math.sqrt(D)
    sink_vec = sink.view(-1)  # (H*q_mul,)
    outs = []
    for b in range(B):
        qb = q[b]  # (T,H,1,D)
        kb = k[b].squeeze(2)  # (T,H,D)
        vb = v[b].squeeze(2)
        full = sdpa(qb, kb, vb, sink_vec, sm_scale, sliding_window)  # (T, H*D)
        if mode == "decode":
            full = full[-1:, :]
        outs.append(full)
    ref = torch.stack(outs, dim=0)  # (B, T_or1, H*D)
    return ref


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """This is reference implementation for sink attention"""
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def _create_sink_tensor(n_heads, dtype, sink_scale, sink_size, q_mul):
    return torch.full((sink_size, n_heads * q_mul), sink_scale, dtype=dtype)


def _make_qkv(bs, seqlen, n_heads, head_dim, dtype, q_mul):
    # q: 5D (B,T,H,q_mul,D); k,v: 4D (B,T,H,D) to match write() expectations
    q = torch.randn(bs, seqlen, n_heads, q_mul, head_dim, dtype=dtype)
    k = torch.randn(bs, seqlen, n_heads, head_dim, dtype=dtype)
    v = torch.randn(bs, seqlen, n_heads, head_dim, dtype=dtype)
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


def _resolve_iree_compile(driver_env: str | None):
    driver = driver_env or os.getenv("IREE_HAL_TARGET_DEVICE", "hip")
    hip_target = os.getenv("IREE_HIP_TARGET", "gfx942")
    compile_args: list[str]
    runtime_driver = driver
    if driver == "local":
        runtime_driver = "local-task"
        compile_args = ["--iree-hal-target-backends=llvm-cpu"]
    else:
        compile_args = [f"--iree-hal-target-device={driver}"]
        if driver == "hip":
            compile_args.append(f"--iree-hip-target={hip_target}")
    cpu_like = (
        runtime_driver in ("local-task", "local")
        or "--iree-hal-target-backends=llvm-cpu" in compile_args
    )
    return runtime_driver, compile_args, cpu_like


class PrefillWrapperEager(torch.nn.Module):
    def __init__(
        self,
        pa: PagedAttention,
        block_index: int,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor,
    ):
        super().__init__()
        self.pa = pa
        self.block_index = block_index
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        self.sink = sink

    def forward(self, q, k, v, cache_state, seq_block_ids, mask=None):
        fn_or_result = self.pa.forward_prefill(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=self.block_index,
            attention_kernel="torch",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
            mask=mask,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )
        return fn_or_result


class DecodeWrapperEager(torch.nn.Module):
    def __init__(
        self,
        pa: PagedAttention,
        block_index: int,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor,
    ):
        super().__init__()
        self.pa = pa
        self.block_index = block_index
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        self.sink = sink

    def forward(
        self,
        q_last,
        k_last,
        v_last,
        cache_state,
        seq_block_ids,
        start_positions,
        mask=None,
    ):
        fn_or_result = self.pa.forward_decode(
            q=q_last,
            k=k_last,
            v=v_last,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=self.block_index,
            start_positions=start_positions,
            attention_kernel="torch",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            mask=mask,
            scale=None,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )

        return fn_or_result


def _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype):
    bs, seq_len, n_heads, _, _ = q.shape
    stride = pa.block_seq_stride

    blocks = math.ceil(seq_len / stride)
    # Each batch gets its own disjoint page id range to avoid KV overlap:
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

    wrapper = PrefillWrapperEager(pa, 0, n_heads, sliding_window, sink)
    prefill = wrapper(q, k, v, cache_state, seq_block_ids)

    if mode == "prefill":
        return prefill

    else:

        past_len = seq_len - 1
        start_positions = torch.full((bs,), past_len, device=q.device, dtype=torch.long)

        # Single-step decode: feed only the final timestep (shape B,1,...).
        q_last = q[:, -1:, ...]  # (B,1,H,M,D)
        k_last = k[:, -1:, ...]  # (B,1,H,D)
        v_last = v[:, -1:, ...]  # (B,1,H,D)

        seq_lens = torch.full((bs,), seq_len, device=q.device, dtype=torch.long)

        decode_mask = decode_attention_mask(
            seq_lens,
            seq_block_ids.shape[1] * pa.block_seq_stride,
            dtype,
            q.device,
        ).to(q.device)

        wrapper = DecodeWrapperEager(pa, 0, n_heads, sliding_window, sink)
        out = wrapper(
            q_last,
            k_last,
            v_last,
            cache_state,
            seq_block_ids,
            start_positions,
            mask=decode_mask,
        )

        return out


class TestPagedAttentionForwardSinkEager:
    @pytest.mark.parametrize(("dtype", "atol", "rtol"), _DT_CASES)
    @pytest.mark.parametrize("sliding_window", _SLIDING_WINDOWS)
    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize(("bs", "seqlen", "n_heads", "head_dim"), _SHAPE_CASES)
    @pytest.mark.parametrize("sink_scale", _SINK_SCALE)
    @pytest.mark.parametrize("context_len", _CONTEXT_LEN)
    @pytest.mark.parametrize("q_mul", _Q_MUL)
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
        head_dim,
        sink_scale,
        context_len,
        q_mul,
    ):
        torch.manual_seed(1234)
        pa = PagedAttention(
            transformer_block_count=1,
            attn_head_count=n_heads,
            attn_head_dim=head_dim,
            attn_type="gqa",
            cache_partition_count=2,
            block_seq_stride=16,
            cache_dtype=dtype,
            attn_dtype=dtype,
            device=None,
        )
        q, k, v = _make_qkv(bs, seqlen, n_heads, head_dim, dtype, q_mul)
        sink = _create_sink_tensor(n_heads, dtype, sink_scale, 1, q_mul)

        out = _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype)
        ref = _reference(q, k, v, sink, mode, sliding_window).to(out.dtype)

        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
