# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from amdsharktank.layers.configs.llm_configs import LlamaHParams, LlamaModelConfig
from amdsharktank.layers.kv_cache import CacheAllocation
from amdsharktank.layers.paged_attention import build_cache
import unittest
import torch
from iree.turbine import aot
from amdsharktank.layers.paged_llama_attention_block import (
    create_paged_llama_attention_block,
    PagedLlamaAttentionBlock,
    PagedLlamaGQAttentionBlock,
    PagedLlamaMLAttentionBlock,
)
from amdsharktank.layers import (
    PagedAttention,
    PagedGQAttention,
    build_rotary_layer,
)
from amdsharktank.layers.testing import make_llama_attention_block_theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor

from transformers import LlamaConfig
import math
import os
from pathlib import Path
from amdsharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    iree_to_torch,
)
from amdsharktank.utils.export import export_model_mlir
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.testing import TempDirTestBase
import iree.compiler
from iree.turbine.aot import (
    FxProgramsBuilder,
    export as export_fx_programs,
)

logger = get_logger(__name__)


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
        reason="https://github.com/nod-ai/amdshark-ai/issues/684",
    )
    @pytest.mark.skipif(
        torch.__version__ >= (2, 5),
        reason="https://github.com/nod-ai/amdshark-ai/issues/684, error slows down CI",
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
            attention_kernel="torch",
        )

        attn = create_paged_llama_attention_block(
            theta=theta,
            config=config,
            model_arch="llama",
            block_index=self.block_index,
            head_count=self.attention_head_count,
            head_dim=self.attention_head_dim,
            head_count_kv=self.head_count_kv,
            rms_epsilon=self.rms_epsilon,
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
    (
        19,
        0.25,
    ),
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


class PrefillWrapperEager(torch.nn.Module):
    def __init__(
        self,
        pa: PagedAttention,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor | None,
    ):
        super().__init__()
        self.pa = pa
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        if sink is not None:
            self.register_buffer("sink", sink)
        else:
            self.sink = None

    def forward(self, q, k, v, seq_lens, cache_state, seq_block_ids):
        fn_or_result = self.pa.forward_prefill(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )
        return fn_or_result


class PrefillAndDecodeWrapper(torch.nn.Module):
    """Combined prefill (full sequence) + decode last token in one call."""

    def __init__(
        self,
        pa: PagedAttention,
        head_count_attn: int,
        sliding_window: int,
        sink: torch.Tensor | None,
    ):
        super().__init__()
        self.pa = pa
        self.head_count_attn = head_count_attn
        self.sliding_window = sliding_window
        if sink is not None:
            self.register_buffer("sink", sink)
        else:
            self.sink = None

    def forward(self, q, k, v, seq_lens, cache_state, seq_block_ids, start_positions):
        _ = self.pa.forward_prefill(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
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
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            start_positions=start_positions,
            attention_kernel="decomposed",
            head_count_attn=self.head_count_attn,
            cache_quantizer=None,
            fake_quant=False,
            scale=None,
            sliding_window=self.sliding_window,
            sink=self.sink,
        )


def _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype):
    bs, seq_len, n_heads = q.shape[:3]
    stride = pa.kv_cache.block_seq_stride

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

    seq_lens = torch.full((bs,), seq_len, device=q.device, dtype=torch.long)

    if mode == "prefill":
        wrapper = PrefillWrapperEager(pa, n_heads, sliding_window, sink)
        prefill = wrapper(q, k, v, seq_lens, cache_state, seq_block_ids)
        return prefill

    else:
        past_len = seq_len - 1
        start_positions = torch.full((bs,), past_len, device=q.device, dtype=torch.long)

        wrapper = PrefillAndDecodeWrapper(pa, n_heads, sliding_window, sink)
        out = wrapper(q, k, v, seq_lens, cache_state, seq_block_ids, start_positions)

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
        deterministic_random_seed,
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

        kv_cache = build_cache(
            transformer_block_count=1,
            attn_head_count=kv_heads,
            attn_head_dim=head_dim,
            block_seq_stride=16,
            cache_dtype=dtype,
        )
        pa = PagedGQAttention(
            kv_cache=kv_cache,
            transformer_block_index=0,
            attn_dtype=dtype,
            activation_dtype=dtype,
            use_rope=True,
            attention_chunk_size=None,
        )

        sink = _create_sink_tensor(n_heads, dtype, sink_scale)
        q, k, v = _make_qkv(bs, seqlen, n_heads, kv_heads, head_dim, dtype)

        ref = _make_reference_for_case(q, k, v, mode, sliding_window, sink).to(q.dtype)
        out = _run_pa_eager(pa, mode, q, k, v, sink, sliding_window, context_len, dtype)

        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
