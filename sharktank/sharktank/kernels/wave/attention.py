# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *
from amdsharktank.kernels.mlir_kernel import *
from amdsharktank.kernels.wave.utils import get_wave_module_body_asm
from wave_lang.kernel.wave.templates.vanilla_attention import (
    get_bhsd_attention_kernel,
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


__all__ = [
    "wave_bhsd_flash_attention",
]


def get_wave_flash_attention_asm(
    target_function_name: str,
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    dynamic_dims: bool,
    is_causal: bool = False,
    is_custom_mask: bool = False,
) -> str:
    base_attention_func, hyperparams, dynamic_symbols = get_bhsd_attention_kernel(
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
        iree_launch_async=False,
    )
    options = set_default_run_config(options)
    with Context() as ctx:
        base_attention = wave_compile(options, base_attention_func)

    asm = base_attention.asm
    return asm


# Wave Attention Kernels
# Each kernel is put into its own class to create a namespace for it
B = StaticDim.B  # batch_size
H = StaticDim.H  # num_query_heads
M = StaticDim.M  # query_seq_len
N = StaticDim.N  # head_size_kv
K1 = StaticDim.K1  # head_size
K2 = StaticDim.K2  # kv_seq_len

F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)


@mlir_kernel(
    inputs=(
        MLIRTensor[B, H, M, K1, F16],
        MLIRTensor[B, H, K2, K1, F16],
        MLIRTensor[B, H, K2, N, F16],
        MLIRTensor[B, H, M, N, F32],
    ),
    results=(MLIRTensor[B, H, M, N, F32],),
)
def wave_bhsd_flash_attention(q, k, v, c, result=None):
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
    dynamic_dims = False
    is_causal = True
    is_custom_mask = False
    i_type_str = "f16"
    o_type_str = "f32"

    wave_kernel_name = f"wave_flash_attention_{batch_size}_{num_heads}_{q_s}_{v_d}_{i_type_str}_{o_type_str}"

    asm = get_wave_flash_attention_asm(
        wave_kernel_name,
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )

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
