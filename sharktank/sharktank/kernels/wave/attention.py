# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
from sharktank.kernels.wave.utils import get_wave_module_body_asm
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_bhsd_attention_kernel,
)
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    Module,
    Context,
)
import torch


__all__ = [
    "wave_bhsd_flash_attention",
    "wave_prefill_attention"
]


def get_wave_flash_attention_asm(
    target_function_name: str,
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    dynamic_dims: bool,
    is_causal: bool = False,
    is_custom_mask: bool = False,
) -> str:
    (base_attention_func, hyperparams, dynamic_symbols,) = get_bhsd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        func_name=target_function_name,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)
    with Context() as ctx:
        base_attention = wave_compile(options, base_attention_func)

    asm = base_attention.asm
    return asm

def get_wave_prefill_attention_asm(
    target_function_name: str,
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    o_shape: tuple[int],
    dynamic_dims: bool,
) -> str:
    prefill_func, hyperparams, dynamic_symbols, dynamic_symbols_map = get_prefill_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        o_shape,
        dynamic_dims,
    )
    hyperparams.update(get_default_scheduling_params())
    opts = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        func_name=target_function_name,
        compile_to_mlir=True,
    )
    opts = set_default_run_config(opts)
    with Context() as ctx:
        compiled = wave_compile(opts, prefill_func)
    return compiled.asm

# Wave Attention Kernels
# Each kernel is put into its own class to create a namespace for it
B = DynDim.B  # batch_size
H = DynDim.H  # num_query_heads
M = DynDim.M  # query_seq_len
N = StaticDim.N  # head_size_kv
K1 = StaticDim.K1  # head_size
K2 = DynDim.K2  # kv_seq_len
S = DynDim.S

F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)
I32 = Dtype.I32(torch.int32)


@mlir_kernel(
    inputs=(
        MLIRTensor[B, H, M, K1, F16],
        MLIRTensor[B, H, K2, K1, F16],
        MLIRTensor[B, H, K2, N, F16],
        MLIRTensor[B, H, M, N, F32],
    ),
    results=(MLIRTensor[B, H, M, N, F32],),
)
def wave_bhsd_masked_flash_attention(q, k, v, c, result=None):
    batch_size, num_heads, q_s, q_d = q.type.shape
    v_batch_size, num_heads_kv, v_s, v_d = v.type.shape
    shape = AttentionShape(
        batch_size=batch_size,
        num_query_heads=num_heads,
        num_kv_heads=num_heads_kv,
        query_seq_len=q_s,
        head_size_kv=v_d,
        head_size=q_d,
        kv_seq_len=v_s,
    )
    mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    dynamic_dims = True
    is_causal = True
    is_custom_mask = False
    i_type_str = "f16"
    o_type_str = "f32"
    batch_size = batch_size if batch_size >= 0 else "B_dyn"
    num_heads = num_heads if num_heads >= 0 else "H_dyn"
    q_s = q_s if q_s >= 0 else "M_dyn"

    wave_kernel_name = f"wave_masked_flash_attention_{batch_size}_{num_heads}_{q_s}_{v_d}_{i_type_str}_{o_type_str}"

    wave_asm = get_wave_flash_attention_asm(
        wave_kernel_name,
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )

    wave_asm_module = Module.parse(wave_asm)
    wave_asm_body = get_wave_module_body_asm(wave_asm_module)

    mlir_wave_kernel = (
        "\n{% raw %}\n"
        + wave_asm_body
        + "\n{% endraw %}\n"
        + f"""
    util.func private @{{{{kernel_name}}}}(%q : !q, %k : !k, %v : !v, %c : !c) -> !result {{
        %c0 = arith.constant 0 : index
        %b = tensor.dim %q, %c0 : !q
        %c1 = arith.constant 1 : index
        %h = tensor.dim %q, %c1 : !q
        %c2 = arith.constant 2 : index
        %m = tensor.dim %q, %c2 : !q
        %k2 = tensor.dim %k, %c2 : !k
        %result = func.call @{wave_kernel_name}(%q, %k, %v, %c, %b, %h, %m, %k2) : (!q, !k, !v, !c, index, index, index, index) -> !result
        util.return %result : !result
    }}
    """
    )
    mlir = "module {" + mlir_wave_kernel + "}"

    return MLIRSpec(mlir)

@mlir_kernel(
    inputs=(
        MLIRTensor[B, H, M, K1, F16],          # q: [B,H,M,DQ]
        MLIRTensor[B, H, K2, K1, F16],         # k: [B,H_KV,K2,DQ]
        MLIRTensor[B, H, K2, N,   F16],        # v: [B,H_KV,K2,DKV]
        MLIRTensor[S, I32],
        MLIRTensor[S, I32],
        MLIRTensor[B, H, M, N, F32],           # c: [B,H,M,DKV]
    ),
    results=(MLIRTensor[B, H, M, N, F32],),
)
def wave_prefill_attention(q, k, v, offsets, sequence_lengths, c, result=None):
    # grab all the run‐time dims
    batch_size, num_heads, q_s, dq = q.type.shape
    _, _, kv_s, dk = v.type.shape
    # S = number of pages per batch
    S_len = offsets.type.shape[0]

    # build the IREE‐wave descriptor
    shape = AttentionShape(
        batch_size=batch_size,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        query_seq_len=q_s,
        head_size=dq,
        head_size_kv=dk,
        kv_seq_len=kv_s,
        num_seqs=S_len,
        max_seq_len=S_len*16,
    )
    mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    dynamic_dims = True

    i_type_str = "f16"
    o_type_str = "f32"

    # batch_size = batch_size if batch_size >= 0 else "B_dyn"
    # num_heads = num_heads if num_heads >= 0 else "H_dyn"
    # q_s = q_s if q_s >= 0 else "M_dyn"
    
    wave_kernel_name = f"wave_prefill_attention_{batch_size if batch_size >= 0 else "B_dyn"}_{num_heads if num_heads >= 0 else "H_dyn"}_{q_s if q_s >= 0 else "M_dyn"}_{dk}_{i_type_str}_{o_type_str}"

    # compile the wave‐MLIR ahead of time
    wave_asm = get_wave_prefill_attention_asm(
        wave_kernel_name,
        shape,
        mfma_variant,
        (q_s, num_heads, dq),
        (kv_s, num_heads, dq),
        (kv_s, num_heads, dk),
        (q_s, num_heads, dk),
        dynamic_dims,
    )
    wave_mod = Module.parse(wave_asm)
    wave_body = get_wave_module_body_asm(wave_mod)

    # wrap it in a util.func that calls into the compiled wave entry point
    mlir_wave = (
        "\n{% raw %}\n"
        + wave_body
        + "\n{% endraw %}\n"
        + f"""
      util.func private @{{{{kernel_name}}}}(
        %q       : !q,
        %k       : !k,
        %v       : !v,
        %offsets : !offs,
        %lens    : !lens,
        %c       : !c
      ) -> !result {{
        %c0    = arith.constant 0 : index
        %b     = tensor.dim %q, %c0 : !q
        %c1    = arith.constant 1 : index
        %h     = tensor.dim %q, %c1 : !q
        %c2    = arith.constant 2 : index
        %m     = tensor.dim %q, %c2 : !q
        %c3    = arith.constant 0 : index
        %s     = tensor.dim %offsets, %c3 : !offs
        %c4    = arith.constant 3 : index
        %k2    = tensor.dim %k, %c4 : !k
        %res   = func.call @{wave_kernel_name}(
          %q, %k, %v, %offsets, %lens, %c, 
          %b, %h, %m, %k2, %s
        ) : (!q, !k, !v, !offs, !lens, !c, index, index, index, index, index) -> !result
        util.return %res : !result
      }}
    """
    )
    return MLIRSpec("module {" + mlir_wave + "}")