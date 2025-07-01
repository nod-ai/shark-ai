# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Paged attention specifically designed for the decoding step of inference.
"""

from sharktank.kernels.mlir_kernel import mlir_kernel, Dtype, StaticDim, DynDim, MLIRTensor
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    paged_decode_attention_shape,
    get_paged_decode_attention_kernels,
    get_paged_decode_intermediate_arrays_shapes
)
from sharktank.kernels.wave.utils import get_wave_module_body_asm
from iree.turbine.kernel.wave.constraints import MMAType, GenericDot, MMAOperand
from iree.turbine.kernel.wave.utils.general_utils import get_default_scheduling_params
from iree.turbine.kernel.wave.compile import (
    wave_compile,
    WaveCompileOptions,
    set_default_run_config,
)
import torch
from iree.compiler.ir import Module, Context

# TODO: should this be hardcoded
BLOCK_SIZE = 32

# Dimensions are taken from Wave's `templates.paged_decode_attention.phase_0`
S = DynDim.NUM_SEQS # TODO: this represents batch size?
B = StaticDim.NUM_QUERY_HEADS
K1 = StaticDim.QUERY_HEAD_DIM
N_KV = DynDim.NUM_KV_BLOCKS
BH = StaticDim.NUM_KV_HEADS
N = StaticDim.KV_HEAD_DIM
K2 = DynDim.KV_BLOCK_TABLE_LEN
U = StaticDim.NUM_KV_SPLITS

BF16 = Dtype.BF16(torch.bfloat16)
F32 = Dtype.F32(torch.float32)


@mlir_kernel(
    inputs=(
        MLIRTensor[S, B, K1, BF16],
        MLIRTensor[N_KV, BH, K1, BF16],
        MLIRTensor[N_KV, BH, N, BF16],
        MLIRTensor[S, B, N, F32],
    ),
    results=(MLIRTensor[S, B, N, F32]),
)
def paged_decode_attention(q, k, v, output, result=None):
    num_seqs, num_query_heads, query_seq_len, query_head_dim = q.type.shape
    
    assert query_seq_len == 1, "Decode attention can only process a single query token"

    _, num_kv_heads, kv_seq_len, kv_head_dim = k.type.shape
    shape = paged_decode_attention_shape(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=query_head_dim,
        head_size_k=kv_head_dim,
        block_size=BLOCK_SIZE,
        num_seqs=num_seqs,
        kv_lens=kv_seq_len,
    )

    attn_logits_shape, attn_logits_max_shape = get_paged_decode_intermediate_arrays_shapes(shape, KV_SPLITS)
    attn_logits = torch.empty(
        attn_logits_shape,
        dtype=torch.float32,
        device=self.device
    )
    attn_logits_max = torch.empty(
        attn_logits_max_shape,
        dtype=torch.float32,
        device=self.device
    )

    use_multi_head_attention = shape.num_query_heads == shape.num_kv_heads

    if use_multi_head_attention:
        mfma_variant = (
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),
        )
    else:
        mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)

    # TODO: should num_kv_splits be kv_lens?
    # TODO: input and output dtype should be FP4
    # TODO: what should logit_cap be?
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        kv_lens,
        input_dtype=Dtype.F16(torch.float16),
        output_dtype=Dtype.F16(torch.float16),
        mha=use_multi_head_attention,
        logit_cap=0.0,
    )

    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=False,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        waves_per_eu=2,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    with Context() as _:
        phase_0 = wave_compile(options, phase_0)

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=False,
        use_buffer_load_ops=False,
        use_buffer_store_ops=False,
        waves_per_eu=4,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    with Context() as _:
        phase_1 = wave_compile(options, phase_1)

    asm_module = Module.parse(asm)
    asm_body = get_wave_module_body_asm(asm_module)

    mlir_wave_kernel = (
        asm_body
        + f"""
    util.func private @{{{{kernel_name}}}}(%q : !q, %k : !k, %v : !v, %c : !c) -> !result {{
        %result = func.call @{wave_kernel_name}(%q, %k, %v, %c) : (!q, !k, !v, !c) -> !result
        util.return %result : !result
    }}
    """
    )
    mlir = "module {" + mlir_wave_kernel + "}"

    return MLIRSpec(mlir)
 