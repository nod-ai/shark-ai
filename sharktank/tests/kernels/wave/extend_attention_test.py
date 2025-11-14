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
from sharktank.utils.testing import is_mi300x, IreeFlags
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
from sharktank.layers.paged_attention import PagedGQAttention
from sharktank.layers.paged_attention import build_cache
from sharktank.utils.testing import assert_tensor_close


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA/HIP device.")
    @pytest.mark.parametrize(
        "batch, heads, seq_len, head_dim, attn_dtype, device",
        [
            (1, 4, 32, 16, torch.float16, "cuda"),
        ],
    )
    def test_extend_kv_cache_single_request_two_chunks(
        self,
        batch,
        heads,
        seq_len,
        head_dim,
        attn_dtype,
        device,
    ):
        """Test extend attention over two sequential chunks for a single request."""
        torch.manual_seed(42)

        transformer_block_count = 8
        transformer_block_index = 3
        block_seq_stride = 4
        chunk_size = 16  # tokens per chunk
        num_chunks = seq_len // chunk_size

        # Full QKV for reference
        q_full = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )
        k_full = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )
        v_full = torch.randn(
            batch, seq_len, heads, head_dim, dtype=attn_dtype, device=device
        )

        # Full SDPA reference (no KV cache)
        q_sdpa = q_full.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)
        input_mask = ops.input_mask(torch.tensor([seq_len]), seq_len)
        a = ops.attention_mask(
            input_mask,
            source_len=seq_len,
            target_len=seq_len,
            attention_dtype=attn_dtype,
        ).to(device)
        sdpa_ref = ops.scaled_dot_product_attention(q=q_sdpa, k=k_sdpa, v=v_sdpa, a=a)

        # Build KV cache
        page_count = batch * seq_len // block_seq_stride
        kv_cache_extend = build_cache(
            transformer_block_count=transformer_block_count,
            attn_head_count=heads,
            attn_head_dim=head_dim,
            block_seq_stride=block_seq_stride,
            cache_dtype=attn_dtype,
            extend_attention=True,
        )

        cache = PagedGQAttention(
            kv_cache=kv_cache_extend,
            transformer_block_index=transformer_block_index,
            attn_dtype=attn_dtype,
            use_rope=True,
            attention_chunk_size=None,
        )

        allocation = cache.allocate(page_count=page_count)
        page_ids = torch.arange(page_count, dtype=torch.int64).view(
            batch, seq_len // block_seq_stride
        )

        wave_kv_cache = kv_cache_extend.unflatten_page_table(allocation).flatten(0, 3)

        # Loop through chunks simulating progressive prefill
        all_outputs = []
        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = (chunk_id + 1) * chunk_size

            q = q_full[:, start:end, :, :]
            k = k_full[:, start:end, :, :]
            v = v_full[:, start:end, :, :]

            write_page_ids = page_ids[
                :, start // block_seq_stride : end // block_seq_stride
            ]
            cache_partitions = [k.cpu(), v.cpu()]

            # Write chunk to KV cache
            cache.write(
                allocation,
                cache_partitions=cache_partitions,
                transformer_block_index=transformer_block_index,
                page_ids=write_page_ids,
            )

            # Create indices for extend_attention kernel
            if start == 0:
                page_ids_prefix = torch.full((page_ids.size(0), 1), 0)
            else:
                page_ids_prefix = page_ids[:, : (start // block_seq_stride)]

            k_indices, v_indices = create_kv_indices(
                page_ids=page_ids_prefix.to(device),
                transformer_block_count=transformer_block_count,
                transformer_block_index=transformer_block_index,
                block_seq_stride=block_seq_stride,
                cache_partitions=cache_partitions,
                dtype=torch.int32,
                device=device,
            )

            seq_lens = torch.tensor([end], dtype=torch.int32, device=device)
            start_positions = torch.tensor([start], dtype=torch.int32, device=device)

            # Call extend_attention on current chunk
            extend_attention_out = ops.extend_attention(
                q=q,
                k=k,
                v=v,
                kv_cache=wave_kv_cache,
                k_indices=k_indices.flatten(),
                v_indices=v_indices.flatten(),
                page_ids=write_page_ids.to(device),
                start_positions=start_positions,
                seq_lens=seq_lens,
            )

            all_outputs.append(extend_attention_out)

            # Compare new chunk output to reference SDPA output for that range
            torch.testing.assert_close(
                extend_attention_out,
                sdpa_ref[:, :, start:end, :],
                atol=1e-3,
                rtol=1e-3,
                msg=f"Mismatch in chunk {chunk_id}",
            )

        # Combine outputs for completeness check
        combined_output = torch.cat(all_outputs, dim=2)
        torch.testing.assert_close(
            combined_output,
            sdpa_ref,
            atol=1e-3,
            rtol=1e-3,
            msg="Combined output does not match full SDPA reference",
        )
