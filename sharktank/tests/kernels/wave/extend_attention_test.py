# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
import pytest
import torch
from iree.compiler.passmanager import PassManager
from iree.compiler.ir import Context, Module
import iree.turbine.aot as aot
from sharktank.kernels.wave.extend_attention import wave_extend_attention
from parameterized import parameterized
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
import iree.compiler as ireec
import iree.runtime as ireert
from pathlib import Path
import numpy as np
from sharktank.utils.testing import is_mi300x, IreeFlags, BatchInput
from sharktank.kernels.wave.utils import (
    create_extend_attention_inputs,
    ref_extend_attn,
    create_kv_indices,
)
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from dataclasses import replace
from torch.testing import assert_close
from sharktank import ops
from sharktank.ops import attention_impls
from sharktank.utils.testing import (
    assert_tensor_close,
    make_random_tokens,
    make_seq_block_ids,
    make_task_inputs_per_request,
    batch_tasks_by_chunk,
)
from sharktank.models.llama.toy_llama import generate
from sharktank.models.llm.llm import PagedLlmModelV1
from sharktank.layers.kv_cache import CacheAllocation


@is_mi300x
@pytest.mark.usefixtures("iree_flags")
class TestExtendAttention:
    def hip_flags(self):
        return [
            f"--iree-hip-target={self.iree_hip_target}",
            f"--iree-hal-target-device={self.iree_hal_target_device}",
            "--iree-opt-level=O3",
            "--iree-dispatch-creation-propagate-collapse-across-expands=true",
            "--iree-codegen-enable-default-tuning-specs=true",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hip-specialize-dispatches",
            "--iree-hal-memoization=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
            "--iree-dispatch-creation-enable-early-trunc-fusion=true",
        ]

    # Wave extend attention kernel requires torch version >= 2.6 to run both eager and export
    # since wave's minimum torch version is 2.6.
    @pytest.mark.skipif(
        torch.__version__ < (2, 6),
        reason="Wave extend attention kernel requires torch version >= 2.6",
    )
    @pytest.mark.parametrize(
        "context_len, num_seqs, num_query_heads, head_size, num_kv_heads, head_size_kv, is_causal",
        [
            (1024, 2, 16, 128, 1, 128, True),
            (2048, 4, 128, 128, 8, 128, True),
        ],
    )
    def test_extend_attention_export_compile_run(
        self,
        iree_flags: IreeFlags,
        tmp_path: Path,
        context_len: int,
        num_seqs: int,
        num_query_heads: int,
        head_size: int,
        num_kv_heads: int,
        head_size_kv: int,
        is_causal: bool,
    ):
        class WaveExtendAttentionModule(torch.nn.Module):
            def forward(
                self,
                q_extend,
                k_extend,
                v_extend,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                k_indices,
                v_indices,
                output,
                max_len_extend_tensor,
            ):
                return wave_extend_attention(
                    q_extend,
                    k_extend,
                    v_extend,
                    k_cache,
                    v_cache,
                    qo_indptr,
                    kv_indptr,
                    k_indices,
                    v_indices,
                    output,
                    max_len_extend_tensor,
                )

        # Use create_inputs from Wave
        shape = AttentionShape(
            context_len=context_len,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size_kv=head_size_kv,
            head_size=head_size,
        )
        dtype = torch.float16
        torch.manual_seed(0)
        (
            q_extend,
            k_extend,
            v_extend,
            k_cache,
            v_cache,
            b_req_idx,
            b_seq_len,
            qo_indptr,
            kv_indptr,
            k_indices,
            v_indices,
            custom_mask,
            mask_offsets,
            b_start_loc,
            b_seq_len_prefix,
            extend_token_num,
            max_len_extend_wave,
            logit_cap,
            _,
            _,
        ) = create_extend_attention_inputs(shape, dtype)
        shape = replace(shape, max_seq_len=max_len_extend_wave)
        output = torch.empty(
            extend_token_num,
            shape.num_query_heads,
            shape.head_size,
            dtype=dtype,
            device=q_extend.device,
        )

        mlir_inputs = (
            q_extend,
            k_extend,
            v_extend,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            k_indices,
            v_indices,
            output,
            torch.tensor(
                max_len_extend_wave, dtype=torch.int32, device=q_extend.device
            ),
        )
        e = aot.export(
            WaveExtendAttentionModule(),
            args=mlir_inputs,
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        assert "func.func @main" in mlir_asm
        assert f"stream.executable private @extend_attention" in mlir_asm
        assert f"func.func private @wave_extend_attention" in mlir_asm
        assert f"util.func private @wave_extend_attention" in mlir_asm
        mlir_path = tmp_path / "wave_extend_attention.mlir"
        with open(str(mlir_path), "w") as f:
            f.write(mlir_asm)
        vmfb = ireec.compile_file(
            str(mlir_path),
            extra_args=self.hip_flags(),
        )

        instance = ireert.VmInstance()
        devices = [ireert.get_device(iree_flags.iree_device)]
        config = ireert.Config(device=devices[0])
        hal = ireert.create_hal_module(instance, devices=devices)
        binary = ireert.VmModule.copy_buffer(instance, vmfb)
        modules = ireert.load_vm_modules(hal, binary, config=config)

        _wave_extend_attention_main = modules[-1].main
        device = torch.device("cpu")
        mlir_inputs = [x.to(device) for x in mlir_inputs]
        iree_results = _wave_extend_attention_main(*mlir_inputs)
        iree_results = torch.from_numpy(
            np.asarray(iree_results.to_host()).astype(np.float32)
        )
        ref_output = ref_extend_attn(
            q_extend=q_extend,
            k_buffer=k_cache,
            v_buffer=v_cache,
            b_req_idx=b_req_idx,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            b_seq_len_prefix=b_seq_len_prefix,
            max_len_extend=max_len_extend_wave,
            extend_token_num=extend_token_num,
            dtype=dtype,
            is_causal=is_causal,
            logit_cap=logit_cap,
        ).cpu()

        assert_close(iree_results, ref_output, rtol=1e-3, atol=1e-3, check_dtype=False)


@is_mi300x
class TestOpsExtendAttention:
    """Test extend attention implementation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA/HIP device.")
    @pytest.mark.parametrize(
        "batch, heads, seq_len, head_dim, attn_dtype, device",
        [
            (1, 8, 128, 32, torch.float16, "cuda"),
            (1, 32, 13, 128, torch.float16, "cuda"),
            (1, 4, 32, 64, torch.float16, "cuda"),
        ],
    )
    def test_no_cache(
        self,
        batch,
        heads,
        seq_len,
        head_dim,
        attn_dtype,
        device,
    ):
        """Test extend attention with various configurations."""
        torch.manual_seed(42)
        q = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )
        k = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )
        v = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )

        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        input_mask = ops.input_mask(torch.tensor([seq_len]), seq_len)
        a = ops.attention_mask(
            input_mask,
            source_len=seq_len,
            target_len=seq_len,
            attention_dtype=attn_dtype,
        ).to(device)
        sdpa = ops.scaled_dot_product_attention(q=q_sdpa, k=k_sdpa, v=v_sdpa, a=a)

        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        start_positions = torch.tensor([0], dtype=torch.int32)
        indices_no_cache = torch.zeros(q.shape[0], dtype=torch.int32)
        extend_attention = ops.extend_attention(
            q=q,
            k=k,
            v=v,
            kv_cache=None,
            k_indices=indices_no_cache,
            v_indices=indices_no_cache,
            start_positions=start_positions,
            seq_lens=seq_lens,
        )
        torch.testing.assert_close(sdpa, extend_attention, atol=1e-3, rtol=1e-3)

        k_noise = k * 0.05
        extend_attention_k_noise = ops.extend_attention(
            q=q,
            k=k_noise,
            v=v,
            kv_cache=None,
            k_indices=indices_no_cache,
            v_indices=indices_no_cache,
            start_positions=start_positions,
            seq_lens=seq_lens,
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                sdpa, extend_attention_k_noise, atol=1e-3, rtol=1e-3
            )

        v_noise = v * 0.05
        extend_attention_v_noise = ops.extend_attention(
            q=q,
            k=k,
            v=v_noise,
            kv_cache=None,
            k_indices=indices_no_cache,
            v_indices=indices_no_cache,
            start_positions=start_positions,
            seq_lens=seq_lens,
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                sdpa, extend_attention_v_noise, atol=1e-3, rtol=1e-3
            )


# @is_mi300x
class TestPrefillExtendAttention:
    """Test prefill extend attention implementation."""

    def setup_sdpa_inputs(
        self,
        model: PagedLlmModelV1,
        seq_lens: torch.Tensor,
        num_requests: int,
        block_seq_stride: int,
    ):
        token_ids = make_random_tokens(model.hp.vocab_size, num_requests, seq_lens)
        start_positions = None
        max_seq_len = torch.max(seq_lens)
        seq_block_ids = make_seq_block_ids(block_seq_stride, num_requests, max_seq_len)
        page_count = num_requests * max_seq_len // block_seq_stride
        sdpa_cache_state = model.cache.allocate(page_count=page_count)
        return (
            token_ids,
            start_positions,
            seq_block_ids,
            page_count,
            sdpa_cache_state,
        )

    def get_sdpa_prefill_logits(
        self,
        model: PagedLlmModelV1,
        token_ids: torch.Tensor,
        start_positions: torch.Tensor | None,
        seq_lens: torch.Tensor,
        seq_block_ids: torch.Tensor,
        cache_state: CacheAllocation,
    ) -> torch.Tensor:
        """Compute logits using standard prefill (non-extend-attention)."""

        logits = model.prefill(
            token_ids,
            start_positions=start_positions,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )
        return logits

    def get_extend_prefill_logits(
        self,
        model: PagedLlmModelV1,
        token_ids: torch.Tensor,
        start_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_block_ids: torch.Tensor,
        cache_state: CacheAllocation,
    ) -> torch.Tensor:
        """Compute logits using extend-attention prefill, splitting token_ids into chunks."""
        extend_attention_logits = model.prefill(
            token_ids,
            seq_lens=seq_lens,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )
        return extend_attention_logits

    def run_extend_attention_chunk(
        self,
        model: PagedLlmModelV1,
        invocation: BatchInput,
        cache_state: CacheAllocation,
        device: torch.device,
    ) -> torch.Tensor:
        """Runs one chunk of extend attention."""
        token_ids = invocation.batch_input_tokens.to(device)
        start_positions = invocation.batch_start_positions.to(device)
        seq_lens = invocation.batch_seq_lens.to(device)
        seq_block_ids = invocation.batch_page_ids.to(device)

        return self.get_extend_prefill_logits(
            model,
            token_ids,
            start_positions,
            seq_lens,
            seq_block_ids,
            cache_state,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA/HIP device.")
    @pytest.mark.parametrize(
        "num_requests, heads, seq_len, head_dim, attn_dtype, device",
        [
            (1, 8, 32, 32, torch.float16, "cuda"),
        ],
    )
    def test_single_request_aligned(
        self,
        num_requests,
        heads,
        seq_len,
        head_dim,
        attn_dtype,
        device,
    ):
        seed = 42
        torch.manual_seed(seed)

        theta, config = generate(seed)
        config.block_seq_stride = 4
        config.use_extend_attention = False
        sdpa_model = PagedLlmModelV1(theta, config)

        r1_seq_lens = torch.tensor([seq_len])
        (
            r1_token_ids,
            start_positions,
            r1_seq_block_ids,
            page_count,
            sdpa_cache_state,
        ) = self.setup_sdpa_inputs(
            sdpa_model, r1_seq_lens, num_requests, config.block_seq_stride
        )

        sdpa_logits = self.get_sdpa_prefill_logits(
            sdpa_model,
            r1_token_ids,
            start_positions,
            r1_seq_lens,
            r1_seq_block_ids,
            sdpa_cache_state,
        )

        config.use_extend_attention = True
        config.device = torch.device(device)
        extend_attn_model = PagedLlmModelV1(theta, config)
        chunk_size = 16
        task_inputs = make_task_inputs_per_request(
            "R1",
            r1_token_ids,
            r1_seq_block_ids,
            chunk_size,
            config.block_seq_stride,
            device,
        )

        batched_tasks = batch_tasks_by_chunk(task_inputs, chunk_size)
        extend_attn_cache_state = extend_attn_model.cache.allocate(
            page_count=page_count
        )
        extend_attn_cache_state.allocation[0] = extend_attn_cache_state.allocation[
            0
        ].to(device)

        prefill_extend_logits = []
        for invocation in batched_tasks:
            extend_attn_logits = self.run_extend_attention_chunk(
                extend_attn_model, invocation, extend_attn_cache_state, device
            )
            prefill_extend_logits.append(extend_attn_logits.cpu())

        all_prefill_extend_logits = torch.cat(prefill_extend_logits, dim=1)
        torch.allclose(sdpa_logits, all_prefill_extend_logits, atol=6e-2, rtol=6e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA/HIP device.")
    @pytest.mark.parametrize(
        "num_requests, heads, r1_seq_len, r2_seq_len, head_dim, attn_dtype, device",
        [
            (2, 8, 250, 300, 32, torch.float16, "cuda"),
        ],
    )
    def test_multiple_requests_ragged(
        self,
        num_requests,
        heads,
        r1_seq_len,
        r2_seq_len,
        head_dim,
        attn_dtype,
        device,
    ):
        seed = 42
        torch.manual_seed(seed)

        theta, config = generate(seed)
        config.block_seq_stride = 32
        config.use_extend_attention = False
        sdpa_model = PagedLlmModelV1(theta, config)

        (
            r1_seq_lens,
            token_ids,
            r1_start_positions,
            r1_seq_block_ids,
            page_count,
            r1_sdpa_cache_state,
        ) = self.setup_sdpa_inputs(
            sdpa_model, r1_seq_len, num_requests, config.block_seq_stride
        )
        breakpoint()
        r2_seq_lens = torch.tensor([r2_seq_len])
        r2_token_ids = make_random_tokens(sdpa_model.hp.vocab_size, 1, r2_seq_lens)
        start_positions = None
        r1_seq_block_ids = make_seq_block_ids(config.block_seq_stride, 1, seq_len)
        r1_page_count = 1 * seq_len // config.block_seq_stride
        sdpa_cache_state = sdpa_model.cache.allocate(page_count=page_count)

        r1_sdpa_logits = self.get_sdpa_prefill_logits(
            sdpa_model,
            r1_token_ids,
            start_positions,
            r1_seq_lens,
            r1_seq_block_ids,
            sdpa_cache_state,
        )

        r2_sdpa_logits = self.get_sdpa_prefill_logits(
            sdpa_model,
            r2_token_ids,
            start_positions,
            r2_seq_lens,
            r2_seq_block_ids,
            sdpa_cache_state,
        )
        breakpoint()

        config.use_extend_attention = True
        config.device = torch.device(device)
        extend_attn_model = PagedLlmModelV1(theta, config)
        chunk_size = 16
        task_inputs = make_task_inputs_per_request(
            "R1",
            r1_token_ids,
            r1_seq_block_ids,
            chunk_size,
            config.block_seq_stride,
            device,
        )

        # task_groups -> function to simulate shortfin batching
        batched_tasks = batch_tasks_by_chunk(task_inputs, chunk_size)
        extend_attn_cache_state = extend_attn_model.cache.allocate(
            page_count=page_count
        )
        extend_attn_cache_state.allocation[0] = extend_attn_cache_state.allocation[
            0
        ].to(device)

        prefill_extend_logits = []
        for invocation in batched_tasks:
            token_ids = invocation.batch_input_tokens.to(device)
            start_positions = invocation.batch_start_positions.to(device)
            seq_lens = invocation.batch_seq_lens.to(device)
            seq_block_ids = invocation.batch_page_ids.to(device)
            extend_attn_logits = self.get_extend_prefill_logits(
                extend_attn_model,
                token_ids,
                start_positions,
                seq_lens,
                seq_block_ids,
                extend_attn_cache_state,
            )
            prefill_extend_logits.append(extend_attn_logits.cpu())

        all_prefill_extend_logits = torch.cat(prefill_extend_logits, dim=1)
        torch.allclose(sdpa_logits, all_prefill_extend_logits, atol=6e-2, rtol=6e-2)
