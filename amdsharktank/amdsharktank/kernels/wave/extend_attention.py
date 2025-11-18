# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from amdsharktank.kernels.base import *
from amdsharktank.kernels.mlir_kernel import *
from amdsharktank.kernels.wave.utils import (
    get_wave_module_body_asm,
    mangle,
)
from amdsharktank.kernels.wave.templates.extend_attention_kernel import (
    get_extend_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    Module,
    Context,
)
import torch
from dataclasses import replace


__all__ = [
    "wave_extend_attention",
]


def get_wave_extend_attention_asm(
    target_function_name: str,
    kernel_params: dict,
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    enable_scheduling: SchedulingType,
    q_extend_shape: tuple[int],
    k_extend_shape: tuple[int],
    v_extend_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.float32,
    size_dtype: torch.dtype = torch.int32,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    layer_scaling: Optional[float] = None,
    num_waves: int = 4,
    use_custom_mask: bool = False,
) -> str:

    (extend_attention, hyperparams, dynamic_symbols,) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend_shape,
        k_extend_shape,
        v_extend_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        is_causal=is_causal,
        logit_cap=logit_cap,
        num_waves=num_waves,
        use_custom_mask=use_custom_mask,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        use_buffer_ops=True,
        func_name=target_function_name,
        compile_to_mlir=True,
        iree_launch_async=False,
    )
    options = set_default_run_config(options)

    with Context() as ctx:
        name = mangle("extend_attention", **kernel_params)
        extend_attention._name = name
        extend_attention = wave_compile(options, extend_attention)

    asm = extend_attention.asm
    return asm


N_Q = DynDim.N_Q
H = StaticDim.H
D_Q = StaticDim.D_Q
N_KV = DynDim.N_KV
H_KV = StaticDim.H_KV
D_KV = StaticDim.D_KV
S = DynDim.S

I32 = Dtype.I32(torch.int32)
F16 = Dtype.F16(torch.float16)


@mlir_kernel(
    inputs=(
        MLIRTensor[N_Q, H, D_Q, F16],
        MLIRTensor[N_KV, H_KV, D_Q, F16],
        MLIRTensor[N_KV, H_KV, D_KV, F16],
        MLIRTensor[N_KV, H_KV, D_Q, F16],
        MLIRTensor[N_KV, H_KV, D_KV, F16],
        MLIRTensor[S, I32],
        MLIRTensor[S, I32],
        MLIRTensor[N_KV, I32],
        MLIRTensor[N_Q, H, D_KV, F16],
        MLIRTensor[I32],
    ),
    results=(MLIRTensor[N_Q, H, D_KV, F16],),
)
def wave_extend_attention(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    out,
    max_seq_len,
    result=None,
):
    n_q, h, d_q = q_extend.type.shape
    n_kv, h_kv, _ = k_extend.type.shape
    _, _, d_kv = v_extend.type.shape
    (s,) = qo_indptr.type.shape
    shape = AttentionShape(
        num_query_heads=h,
        num_kv_heads=h_kv,
        query_seq_len=n_q,
        head_size_kv=d_kv,
        head_size=d_q,
        kv_seq_len=n_kv,
        max_seq_len=n_kv,  # TODO: figure out how to pass in int max_seq_len to use in wave kernel
    )
    mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    n_q = n_q if n_q >= 0 else "N_Q_dyn"
    n_kv = n_kv if n_kv >= 0 else "N_KV_dyn"
    s = s if s >= 0 else "S_dyn"
    qkv_i_type_str = "f16"
    indices_i_type_str = "i32"
    # TODO: don't hardcode the output type, should be dynamic based on the kv-cache-dtype
    o_type_str = "f16"
    kernel_params = {
        N_Q.name: n_q,
        H.name: h,
        D_Q.name: d_q,
        N_KV.name: n_kv,
        H_KV.name: h_kv,
        D_KV.name: d_kv,
        S.name: s,
        "qkv_input_dtype": qkv_i_type_str,
        "indices_input_dtype": indices_i_type_str,
        "output_dtype": o_type_str,
    }
    name = mangle("wave_extend_attention", **kernel_params)
    wave_kernel_fn_name = name

    wave_asm = get_wave_extend_attention_asm(
        wave_kernel_fn_name,
        kernel_params,
        shape,
        mfma_variant,
        SchedulingType.NONE,
        q_extend.type.shape,
        k_extend.type.shape,
        v_extend.type.shape,
        k_buffer.type.shape,
        v_buffer.type.shape,
        out.type.shape,
        torch.float16,
        torch.float16,
        is_causal=True,
    )

    wave_asm_module = Module.parse(wave_asm)
    wave_asm_body = get_wave_module_body_asm(wave_asm_module)

    mlir_wave_kernel = (
        "\n{% raw %}\n"
        + wave_asm_body
        + "\n{% endraw %}\n"
        + f"""
    util.func private @{{{{kernel_name}}}}(%q_extend : !q_extend, %k_extend : !k_extend, %v_extend : !v_extend, %k_buffer : !k_buffer, %v_buffer : !v_buffer, %qo_indptr : !qo_indptr, %kv_indptr : !kv_indptr, %kv_indices : !kv_indices, %out : !out, %max_seq_len : !max_seq_len) -> !result {{
        %max_seq_len_i32 = tensor.extract %max_seq_len[] : tensor<i32>
        %result = func.call @{wave_kernel_fn_name}(%q_extend, %k_extend, %v_extend, %k_buffer, %v_buffer, %qo_indptr, %kv_indptr, %kv_indices, %out, %max_seq_len_i32) : (!q_extend, !k_extend, !v_extend, !k_buffer, !v_buffer, !qo_indptr, !kv_indptr, !kv_indices, !out, i32) -> !result
        util.return %result : !result
    }}
    """
    )

    mlir = "module {" + mlir_wave_kernel + "}"

    return MLIRSpec(mlir)
