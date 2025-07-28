# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
from sharktank.kernels.wave.utils import get_wave_module_body_asm
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_bhsd_attention_kernel,
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
    "wave_bhsd_masked_flash_attention",
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
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    # dynamic_dims: bool,
    is_causal: bool = False,
    # is_custom_mask: bool = False,
) -> str:
    # print("hello", q_shape, k_shape, v_shape, k_cache_shape, v_cache_shape, o_shape)
    (base_prefill_func, hyperparams, dynamic_symbols,) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        # dynamic_dims,
        is_causal=is_causal,
        # is_custom_mask=is_custom_mask,
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
        base_prefill = wave_compile(options, base_prefill_func)

    asm = base_prefill.asm
    return asm


# Wave Attention Kernels
# Each kernel is put into its own class to create a namespace for it
B = DynDim.B  # batch_size
H = DynDim.H  # num_query_heads

M = DynDim.M  # query_seq_len
N = StaticDim.N  # head_size_kv
K1 = StaticDim.K1  # head_size
K2 = DynDim.K2  # kv_seq_len

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

N_Q = DynDim.N_Q
N_KV = DynDim.N_KV
S = DynDim.S

H = StaticDim.H
H_KV = StaticDim.H_KV

D_Q = StaticDim.D_Q
D_KV = StaticDim.D_KV

        # q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype, q_layout],
        # k: tkl.Memory[N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_layout],
        # v: tkl.Memory[N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_layout],
        # k_cache: tkl.Memory[
        #     N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_cache_layout
        # ],
        # v_cache: tkl.Memory[
        #     N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_cache_layout
        # ],

@mlir_kernel(
    inputs=(
        MLIRTensor[N_Q, H, D_Q, F16],      # q_extend
        MLIRTensor[N_KV, H_KV, D_Q, F16],  # k_extend  
        MLIRTensor[N_KV, H_KV, D_KV, F16], # v_extend
        MLIRTensor[N_KV, H_KV, D_Q, F16],  # k_cache (full buffer)
        MLIRTensor[N_KV, H_KV, D_KV, F16], # v_cache (full buffer)
        MLIRTensor[S, I32],                # qo_indptr
        MLIRTensor[S, I32],                # kv_indptr
        MLIRTensor[N_KV, I32],             # kv_indices
        MLIRTensor[I32],                   # max_seq_len
        MLIRTensor[N_Q, H, D_KV, F32],     # output
    ),
    results=(MLIRTensor[N_Q, H, D_KV, F32],),
)
def wave_prefill_attention(q_extend, k_extend, v_extend, k_cache, v_cache, 
                          qo_indptr, kv_indptr, kv_indices, max_seq_len, output, result=None):
    q_s, num_heads, q_d = q_extend.type.shape
    v_s, num_heads_kv, v_d = v_extend.type.shape
    
    shape = AttentionShape(
        num_query_heads=num_heads,
        num_kv_heads=num_heads_kv,
        head_size_kv=v_d,
        head_size=q_d,
        max_seq_len=v_s,
    )
    
    mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    
    # Required parameters for extend attention
    logit_cap = 30.0
    num_waves = 2
    use_custom_mask = False
    is_causal = True
    
    wave_kernel_name = f"wave_prefill_attention_{num_heads}_{q_s if q_s >= 0 else 'M_dyn'}_{v_d}_f16_f32"
    
    (extend_attention_func, hyperparams, dynamic_symbols) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend.type.shape,
        k_extend.type.shape, 
        v_extend.type.shape,
        k_cache.type.shape,
        v_cache.type.shape,
        output.type.shape,
        is_causal=is_causal,
        logit_cap=logit_cap,
        num_waves=num_waves,
        use_custom_mask=use_custom_mask,
    )
    
    hyperparams.update(get_default_scheduling_params())
    
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        func_name=wave_kernel_name,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)
    
    with Context() as ctx:
        extend_attention = wave_compile(options, extend_attention_func)

    wave_asm_module = Module.parse(extend_attention.asm)
    wave_asm_body = get_wave_module_body_asm(wave_asm_module)

    # Fixed MLIR template with correct argument count and order
    mlir_wave_kernel = (
        "\n{% raw %}\n"
        + wave_asm_body
        + "\n{% endraw %}\n"
        + f"""
    util.func private @{{{{kernel_name}}}}(%q_extend : !q_extend, %k_extend : !k_extend, %v_extend : !v_extend, %k_cache : !k_cache, %v_cache : !v_cache, %qo_indptr : !qo_indptr, %kv_indptr : !kv_indptr, %kv_indices : !kv_indices, %max_seq_len : !max_seq_len, %output : !output) -> !result {{
        %c0 = arith.constant 0 : index
        %n_q = tensor.dim %q_extend, %c0 : !q_extend
        %n_kv = tensor.dim %k_cache, %c0 : !k_cache
        %s = tensor.dim %qo_indptr, %c0 : !qo_indptr
        
        %max_len = tensor.extract %max_seq_len[] : !max_seq_len
        
        // Pass all 13 required arguments in the correct order
        %result = func.call @{wave_kernel_name}(%q_extend, %k_extend, %v_extend, %k_cache, %v_cache, %qo_indptr, %kv_indptr, %kv_indices, %output, %max_len, %n_q, %n_kv, %s) : (!q_extend, !k_extend, !v_extend, !k_cache, !v_cache, !qo_indptr, !kv_indptr, !kv_indices, !output, i32, index, index, index) -> !result
        util.return %result : !result
    }}
    """
    )
    
    mlir = "module {" + mlir_wave_kernel + "}"
    return MLIRSpec(mlir)